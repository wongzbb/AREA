#!/usr/bin/env python3
"""
AREA main entry point.

Config precedence (highest first):
  CLI args → per-dataset config.yaml → defaults.yaml

Usage:
  python run.py --config configs/evqa.yaml [--model_name ...] [--k_max 3] ...

Full flag list is printed with --help.
"""
import argparse
import os
import sys
import time
from typing import Optional

import torch
import tqdm
import ujson
import yaml

# Make AREA src importable
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from data import get_dataset, PASSAGE_DELIMITER

# ─── Argument parsing ─────────────────────────────────────────────────────────

def _expand_config_value(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_config_value(item) for key, item in value.items()}
    return value


def load_yaml(path: Optional[str]) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return _expand_config_value(yaml.safe_load(f) or {})


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AREA inference")

    # ── Config files ──────────────────────────────────────────────────────────
    p.add_argument("--config", type=str, default=None,
                   help="Per-dataset YAML config file")
    p.add_argument("--defaults_config", type=str,
                   default=os.path.join(os.path.dirname(__file__), "configs", "defaults.yaml"),
                   help="Defaults YAML config file")

    # ── Dataset / paths ───────────────────────────────────────────────────────
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--query_path", type=str, default=None)
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--wiki_KB", type=str, default=None)
    p.add_argument("--img_index_path", type=str, default=None)
    p.add_argument("--img_index_json_path", type=str, default=None)
    p.add_argument("--output_root", type=str, default="./area_outputs")
    p.add_argument("--experiment_type", type=str, default="with_retrieval",
                   choices=["with_retrieval", "no_retrieval"])

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model_max_length", type=int, default=16384)
    p.add_argument("--min_pixels", type=int, default=3136)
    p.add_argument("--max_pixels", type=int, default=301056)
    p.add_argument("--max_new_tokens", type=int, default=128)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_context_sections", type=int, default=8,
                   help="Cap retrieved passages used in the prompt; <=0 keeps all passages")
    p.add_argument("--eval_passages_w_images", action="store_true")
    p.add_argument("--KB_images", type=str, default=None)
    p.add_argument("--use_google_lens", action="store_true")
    p.add_argument("--crop_query_img", action="store_true")

    # ── AREA hyper-parameters ───────────────────────────────────────────────
    p.add_argument("--k_max", type=int, default=3,
                   help="EAEG: max sentences to highlight")
    p.add_argument("--beta_min", type=float, default=1.5,
                   help="EAEG: minimum bbox scale factor")
    p.add_argument("--beta_max", type=float, default=2.5,
                   help="EAEG: maximum bbox scale factor")
    p.add_argument("--kappa", type=float, default=1.0,
                   help="ETML: entropy-peak z-score threshold")
    p.add_argument("--K_max", type=int, default=4,
                   help="ETML: max re-injection events per generation")
    p.add_argument("--warmup", type=int, default=0,
                   help="DG-LT: samples with gate forced ON before median activates (0 = no warmup needed)")
    p.add_argument("--eps", type=float, default=1e-8)


    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    cli_args = p.parse_args()

    # ── Merge: defaults < dataset config < CLI ────────────────────────────────
    defaults = load_yaml(cli_args.defaults_config)
    dataset_cfg = load_yaml(cli_args.config)
    merged = {**defaults, **dataset_cfg}

    # CLI values override YAML when the flag was explicitly passed, even if the
    # value equals argparse's default. This matters for per-dataset YAML values
    # such as K_max: 0 that a tuning command may intentionally override with
    # --K_max 4.
    option_dest = {
        opt: action.dest
        for action in p._actions
        for opt in action.option_strings
    }
    explicit_cli_dests = set()
    for token in sys.argv[1:]:
        opt = token.split("=", 1)[0]
        if opt in option_dest:
            explicit_cli_dests.add(option_dest[opt])

    cli_dict = vars(cli_args)
    for key, val in cli_dict.items():
        if key in ("config", "defaults_config"):
            continue
        if key in explicit_cli_dests:
            merged[key] = val
        elif key not in merged:
            merged[key] = val  # apply parser default only if YAML didn't set it


    args = argparse.Namespace(**merged)
    return args


# ─── Entity extraction ────────────────────────────────────────────────────────

def extract_entity(query: str) -> Optional[str]:
    """Extract a named or noun-phrase entity from the question."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        if doc.ents:
            return str(doc.ents[-1])
        chunks = list(doc.noun_chunks)
        if chunks:
            return str(chunks[-1])
    except Exception:
        pass
    return None


# ─── Output helpers ───────────────────────────────────────────────────────────

def build_response(sample: dict, prediction: str, info: dict, elapsed: float, dataset_name: str) -> dict:
    base = {
        "data_id": sample.get("data_id", ""),
        "question": sample.get("question", ""),
        "answers": sample.get("answers", [sample.get("answer", "")]),
        "prediction": prediction,
        "elapsed_time": elapsed,
        "gate_txt": info.get("gate_txt", None),
        "gate_vis": info.get("gate_vis", None),
        "k_txt": info.get("k_txt", None),
        "beta": info.get("beta", None),
        "delta_txt": info.get("delta_txt", None),
        "delta_vis": info.get("delta_vis", None),
        "etml_reinjections": info.get("etml_reinjections", None),
        "image_path": sample.get("image_path", ""),
        "image_cropped": info.get("image_cropped", False),
    }
    if "category" in sample:
        base["category"] = sample["category"]
    return base


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = build_args()

    import random
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print("AREA configuration:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    if "Qwen2.5-VL" in args.model_name:
        from area.model import AREAInferenceModelQwen2_5_VL
        model = AREAInferenceModelQwen2_5_VL(args)
    else:
        raise NotImplementedError(
            f"Model {args.model_name!r} not yet supported by AREA. "
            "Currently supported: Qwen/Qwen2.5-VL-*."
        )

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset, start_idx, end_idx = get_dataset(args)
    print(f"Dataset '{args.dataset_name}': {len(dataset)} samples (indices {start_idx}–{end_idx})")

    # ── Load retriever ────────────────────────────────────────────────────────
    from retriever import get_retriever
    retriever = get_retriever(args)

    # ── Output setup ──────────────────────────────────────────────────────────
    part = int(os.environ.get("PART", "0"))
    answer_root = os.path.join(args.output_root, args.dataset_name,
                               os.path.basename(args.model_name))
    os.makedirs(answer_root, exist_ok=True)
    answers_file = os.path.join(answer_root, f"split_{part}.json")
    print(f"Output: {answers_file}")

    # ── Inference loop ────────────────────────────────────────────────────────
    responses = []
    for i, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc=args.dataset_name):
        start_time = time.time()
        query = sample["question"]
        image_query = sample["image_query"]

        # Retrieval
        context_sections = []
        if args.experiment_type == "with_retrieval":
            wiki_url = sample.get("wikipedia_url", "")
            if args.dataset_name == "viquae":
                # ViQuAE's `url` field is the image URL; the Wikipedia page title
                # is carried in output.provenance and exposed by the dataset loader.
                wiki_title = sample.get("wiki_title") or (
                    wiki_url.split("/wiki/")[-1].replace("_", " ") if "/wiki/" in wiki_url else None
                )
                sections, _, _ = retriever.retrieve(image_query, wiki_title=wiki_title)
            elif args.dataset_name in ("evqa", "infoseek"):
                # Pass wikipedia_url so OracleRetriever can use it as fallback
                sections, _, _ = retriever.retrieve(image_query, wikipedia_url=wiki_url)
            else:
                sections, _, _ = retriever.retrieve(image_query)
            max_sections = getattr(args, "max_context_sections", 8)
            if max_sections and max_sections > 0:
                sections = sections[:max_sections]
            context_sections = sections

        context = PASSAGE_DELIMITER.join(context_sections) + ("." if context_sections else "")

        # Entity extraction for visual probe
        entity = sample.get("entity") or extract_entity(query)

        try:
            answer, info = model.area_generate(
                query=query,
                image=image_query,
                context=context,
                entity=entity,
                k_max=args.k_max,
                beta_min=args.beta_min,
                beta_max=args.beta_max,
                kappa=args.kappa,
                K_max=args.K_max,
                warmup=args.warmup,
                eps=args.eps,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"[!] Error on sample {i}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            answer = ""
            info = {}

        elapsed = time.time() - start_time

        if args.verbose:
            print(f"\n[{i}] Q: {query}")
            print(f"     A: {answer}")
            print(f"     gate_txt={info.get('gate_txt')}, gate_vis={info.get('gate_vis')}, "
                  f"k_txt={info.get('k_txt')}, beta={info.get('beta', 0.0):.2f}, "
                  f"elapsed={elapsed:.1f}s")

        responses.append(build_response(sample, answer, info, elapsed, args.dataset_name))

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(answers_file, "w") as f:
        f.write(ujson.dumps(responses, indent=2))
    print(f"\nSaved {len(responses)} predictions → {answers_file}")


if __name__ == "__main__":
    main()
