#!/usr/bin/env python3
"""
AREA evaluation script.

Computes the configured benchmark metrics for each dataset.

Usage:
    python evaluate.py --predictions <path-to-split_*.json> --dataset <name>

For AMBER, also pass --amber_annotation to the annotations.json file.
"""
import argparse
import json
import math
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional


# ─── Text normalisation (shared by E-VQA / InfoSeek / ViQuAE / TextVQA) ──────

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation and extra spaces."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_number(text: str) -> Optional[float]:
    """Try to parse text as a number; return None if not possible."""
    text = text.replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def exact_match_score(prediction: str, references: List[str]) -> float:
    """1.0 if normalised prediction matches ANY reference, else 0.0."""
    pred_n = _normalize(prediction)
    for ref in references:
        if pred_n == _normalize(str(ref)):
            return 1.0
    return 0.0


# ─── Per-dataset evaluators ───────────────────────────────────────────────────

def _token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 (SQuAD-style) between prediction and one reference."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens  = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in set(pred_tokens))
    if common == 0:
        return 0.0
    prec = common / len(pred_tokens)
    rec  = common / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)


def best_f1_score(prediction: str, references: List[str]) -> float:
    """Max token-F1 against any reference."""
    return max((_token_f1(prediction, str(r)) for r in references), default=0.0)


def eval_kb_vqa(predictions: List[Dict]) -> Dict:
    """E-VQA / InfoSeek / ViQuAE — exact-match + token-F1 accuracy."""
    em_correct = f1_total = total = 0
    for r in predictions:
        refs = r.get("answers", [])
        if isinstance(refs, str):
            refs = [refs]
        pred = r.get("prediction", "")
        em_correct += exact_match_score(pred, refs)
        f1_total   += best_f1_score(pred, refs)
        total += 1
    em_acc  = 100.0 * em_correct / total if total else 0.0
    f1_acc  = 100.0 * f1_total   / total if total else 0.0
    return {
        "accuracy":    round(em_acc, 2),   # primary (exact-match)
        "f1_accuracy": round(f1_acc, 2),   # secondary (token-F1)
        "correct": int(em_correct),
        "total": total,
    }


def eval_multiple_choice(predictions: List[Dict]) -> Dict:
    """RealWorldQA / V*Bench — extract first A/B/C/D letter and compare."""
    correct = total = 0
    for r in predictions:
        refs = r.get("answers", [r.get("answer", "")])
        if isinstance(refs, str):
            refs = [refs]
        pred = r.get("prediction", "").strip()
        # Extract option letter from prediction
        m = re.match(r"^\s*\(?([A-Da-d])\)?[\.\):\s]", pred)
        if m:
            pred_letter = m.group(1).upper()
        else:
            # Take first word if it's a single letter
            first = pred.split()[0] if pred.split() else ""
            pred_letter = first.upper() if len(first) == 1 else pred.upper()
        ref_letter = _normalize(refs[0]).strip("().")
        correct += 1 if pred_letter == ref_letter.upper() else 0
        total += 1
    acc = 100.0 * correct / total if total else 0.0
    return {"accuracy": round(acc, 2), "correct": correct, "total": total}


def _normalize_textvqa(text: str) -> str:
    """
    TextVQA official normalisation: lowercase, remove articles, strip punctuation.
    Source: textvqa.org evaluation script.
    """
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)   # remove English articles
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def eval_textvqa(predictions: List[Dict]) -> Dict:
    """
    TextVQA — VQA relaxed accuracy.
    Score for each sample = min(count_matching_references / 3, 1.0)
    where references is the list of 10 human annotations.
    Normalisation includes article removal (a / an / the) per official eval script.
    """
    total_score = 0.0
    total = 0
    for r in predictions:
        refs = r.get("answers", [])
        if isinstance(refs, str):
            refs = [refs]
        pred_n = _normalize_textvqa(r.get("prediction", ""))
        matches = sum(1 for ref in refs if _normalize_textvqa(str(ref)) == pred_n)
        total_score += min(matches / 3.0, 1.0)
        total += 1
    acc = 100.0 * total_score / total if total else 0.0
    return {"vqa_accuracy": round(acc, 2), "total": total}


def _chartqa_try_float(text: str) -> Optional[float]:
    """Parse number from raw string (strip commas only, preserve decimal points)."""
    try:
        return float(str(text).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def eval_chartqa(predictions: List[Dict]) -> Dict:
    """
    ChartQA — relaxed accuracy (±5% for numbers, exact-match for strings).
    Matches official ChartQA evaluation:
      - Numerical: parse raw strings as floats (comma-stripped only), ±5% tolerance.
      - String: simple case-insensitive comparison without punctuation removal.
    Note: _normalize is intentionally NOT used here — it strips decimal points and
    percent signs, causing both false positives and false negatives on chart data.
    """
    correct = total = 0
    for r in predictions:
        refs = r.get("answers", [r.get("answer", "")])
        if isinstance(refs, str):
            refs = [refs]
        pred = r.get("prediction", "").strip()
        pred_num = _chartqa_try_float(pred)

        score = 0
        for ref in refs:
            ref_num = _chartqa_try_float(str(ref))
            if pred_num is not None and ref_num is not None:
                denom = abs(ref_num) if ref_num != 0 else 1e-8
                if abs(pred_num - ref_num) / denom <= 0.05:
                    score = 1
            else:
                if pred.lower().strip() == str(ref).lower().strip():
                    score = 1
            if score:
                break
        correct += score
        total += 1
    acc = 100.0 * correct / total if total else 0.0
    return {"relaxed_accuracy": round(acc, 2), "correct": correct, "total": total}


def _ocrbench_match(prediction: str, references: List[str]) -> float:
    """
    Official OCRBench scoring: substring match after lowercase+strip.
    Score = 1 if ANY ref is in pred OR pred is in ref (case-insensitive).
    Source: github.com/Yuliang-Liu/MultimodalOCR evaluation_ocrbench()
    """
    pred = prediction.lower().strip()
    for ref in references:
        ref = str(ref).lower().strip()
        if ref in pred or pred in ref:
            return 1.0
    return 0.0


def eval_ocrbench(predictions: List[Dict]) -> Dict:
    """
    OCRBench — accuracy per category, overall score = mean(per-category-acc).
    Scoring: official substring match (answer in pred OR pred in answer).
    """
    per_cat: Dict[str, List[float]] = defaultdict(list)
    for r in predictions:
        refs = r.get("answers", [r.get("answer", "")])
        if isinstance(refs, str):
            refs = [refs]
        pred = r.get("prediction", "")
        cat = r.get("category", "Unknown")
        score = _ocrbench_match(pred, [str(x) for x in refs])
        per_cat[cat].append(score)

    cat_scores = {cat: round(100.0 * sum(v) / len(v), 2) for cat, v in per_cat.items()}
    overall = round(sum(cat_scores.values()) / len(cat_scores), 2) if cat_scores else 0.0
    return {"overall_score": overall, "per_category": cat_scores,
            "total": sum(len(v) for v in per_cat.values())}


def eval_pope(predictions: List[Dict]) -> Dict:
    """
    POPE — binary yes/no hallucination benchmark.
    Metrics: Accuracy, Precision, Recall, F1.
    """
    TP = FP = TN = FN = 0
    for r in predictions:
        refs = r.get("answers", [r.get("answer", "")])
        if isinstance(refs, str):
            refs = [refs]
        pred = r.get("prediction", "").lower().strip()
        # Extract yes/no from prediction
        if "yes" in pred.split()[:3] or pred.startswith("yes"):
            pred_label = "yes"
        elif "no" in pred.split()[:3] or pred.startswith("no"):
            pred_label = "no"
        else:
            pred_label = "yes" if "yes" in pred else "no"

        gt = _normalize(str(refs[0]))

        if gt == "yes" and pred_label == "yes":
            TP += 1
        elif gt == "yes" and pred_label == "no":
            FN += 1
        elif gt == "no" and pred_label == "yes":
            FP += 1
        else:
            TN += 1

    total = TP + FP + TN + FN
    acc = 100.0 * (TP + TN) / total if total else 0.0
    prec = 100.0 * TP / (TP + FP) if (TP + FP) else 0.0
    rec = 100.0 * TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "total": total,
    }


def eval_vstar(predictions: List[Dict]) -> Dict:
    """V*Bench — per-category accuracy (Direct Attributes, Relative Position, OCR, GPT4V-Hard)."""
    per_cat: Dict[str, List[float]] = defaultdict(list)
    total_correct = 0
    for r in predictions:
        refs = r.get("answers", [r.get("answer", "")])
        if isinstance(refs, str):
            refs = [refs]
        pred = r.get("prediction", "").strip()
        cat = r.get("category", "all")
        # Extract option letter
        m = re.match(r"^\s*\(?([A-Da-d])\)?[\.\):\s]?", pred)
        if m:
            pred_letter = m.group(1).upper()
        else:
            tok = pred.split()[0] if pred.split() else ""
            pred_letter = tok.upper() if len(tok) == 1 else pred[:1].upper()
        ref_letter = _normalize(str(refs[0])).strip("(). ").upper()
        score = 1.0 if pred_letter == ref_letter else 0.0
        per_cat[cat].append(score)
        total_correct += score

    cat_scores = {cat: round(100.0 * sum(v) / len(v), 2) for cat, v in per_cat.items()}
    total = sum(len(v) for v in per_cat.values())
    overall = round(100.0 * total_correct / total, 2) if total else 0.0
    return {"accuracy": overall, "per_category": cat_scores, "total": total}


def eval_amber_disc(
    predictions: List[Dict],
    annotation_path: str,
) -> Dict:
    """
    AMBER discriminative accuracy.
    Expects annotation_path → datasets/AMBER/data/annotations.json
    """
    with open(annotation_path) as f:
        annotations = json.load(f)
    id2ann = {a["id"]: a for a in annotations}

    correct = total = 0
    per_type: Dict[str, List[float]] = defaultdict(list)

    for r in predictions:
        # data_id format: "AMBER_{id}"
        raw_id = r.get("data_id", "")
        amber_id = int(raw_id.replace("AMBER_", "")) if raw_id.startswith("AMBER_") else None
        if amber_id is None or amber_id not in id2ann:
            continue
        ann = id2ann[amber_id]
        ann_type = ann.get("type", "")
        if "discriminative" not in ann_type:
            continue

        ground_truth = str(ann.get("truth", "")).lower().strip()  # 'yes' or 'no'
        pred = r.get("prediction", "").lower().strip()
        # Extract yes/no
        if "yes" in pred.split()[:3] or pred.startswith("yes"):
            pred_yn = "yes"
        elif "no" in pred.split()[:3] or pred.startswith("no"):
            pred_yn = "no"
        else:
            pred_yn = "yes" if "yes" in pred else "no"

        score = 1.0 if pred_yn == ground_truth else 0.0
        correct += score
        total += 1
        per_type[ann_type].append(score)

    acc = 100.0 * correct / total if total else 0.0
    per_type_scores = {t: round(100.0 * sum(v) / len(v), 2) for t, v in per_type.items()}
    return {
        "amber_d_accuracy": round(acc, 2),
        "per_type": per_type_scores,
        "correct": int(correct), "total": total,
    }


# ─── Dispatcher ───────────────────────────────────────────────────────────────

EVALUATORS = {
    "evqa":          eval_kb_vqa,
    "infoseek":      eval_kb_vqa,
    "viquae":        eval_kb_vqa,
    "real_world_qa": eval_multiple_choice,
    "textvqa":       eval_textvqa,
    "chartqa":       eval_chartqa,
    "ocrbench":      eval_ocrbench,
    "pope":          eval_pope,
    "vstar":         eval_vstar,
    "amber":         None,  # handled separately (needs annotation path)
}


def main():
    p = argparse.ArgumentParser(description="AREA evaluator")
    p.add_argument("--predictions", required=True,
                   help="Path to split_*.json output from run.py "
                        "(glob supported, e.g. 'area_outputs/evqa/**/split_*.json')")
    p.add_argument("--dataset", required=True,
                   choices=list(EVALUATORS.keys()),
                   help="Dataset name")
    p.add_argument("--amber_annotation",
                   default=None,
                   help="AMBER annotations.json path (only needed for amber)")
    args = p.parse_args()

    # Load prediction files (support glob)
    import glob as _glob
    files = sorted(_glob.glob(args.predictions))
    if not files:
        # Try as a literal path
        files = [args.predictions]
    preds: List[Dict] = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        preds.extend(data)

    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset}")
    print(f"Samples : {len(preds)}")
    print(f"{'='*60}")

    if not preds:
        print("No predictions found.")
        sys.exit(1)

    if args.dataset == "amber":
        results = eval_amber_disc(preds, args.amber_annotation)
    else:
        results = EVALUATORS[args.dataset](preds)

    # Pretty print
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        if isinstance(v, dict):
            return "\n  " + "\n  ".join(f"{kk}: {_fmt(vv)}" for kk, vv in sorted(v.items()))
        return str(v)

    for k, v in results.items():
        print(f"  {k}: {_fmt(v)}")
    print()


if __name__ == "__main__":
    main()
