"""
AREA dataset loaders — local-path-aware wrappers.

All paths are taken from args (populated by YAML config + CLI overrides).
Each dataset yields dicts with:
  image_query   : PIL.Image (RGB)
  question      : str
  data_id       : str
  answer        : str | list[str] (for datasets with multiple gold answers)
  image_path    : str | None
  wikipedia_url : str | None  (KB-VQA datasets only)
  entity        : str | None  (optional question-focus entity for vis probe)
"""
import csv
import gzip
import json
import os
import io
from typing import List, Optional, Dict, Any

import torch
from PIL import Image

PASSAGE_DELIMITER = "\n\n\n"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe_image(path: Optional[str]) -> Image.Image:
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), color=(0, 0, 0))


def _download_image(url: str) -> Optional[Image.Image]:
    """Download an image from URL; returns None on failure."""
    try:
        import requests
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return None


def uniform_passages_of_sentences(text: str, n: int = 100) -> List[str]:
    """Split text into ~n-word passages (uses the configured passage format)."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    sentences = list(nlp(text).sents)
    passages, passage, tokens_in_passage = [], [], 0
    for sent in sentences:
        if tokens_in_passage + len(sent) > n:
            if passage:
                passages.append(" ".join(passage))
            passage = [sent.text]
            tokens_in_passage = len(sent)
        else:
            passage.append(sent.text)
            tokens_in_passage += len(sent)
    if passage:
        passages.append(" ".join(passage))
    return passages


def calculate_splits(dataset_len: int):
    part = int(os.environ.get("PART", "0"))
    total_part_env = os.environ.get("TOTAL_PART", None)
    total_part = (int(total_part_env) + 1) if total_part_env is not None else 1
    slicing = dataset_len // total_part
    if (part + 1) == total_part:
        start_idx, end_idx = slicing * part, dataset_len
    else:
        start_idx, end_idx = slicing * part, slicing * part + slicing
    max_samples = int(os.environ.get("MAX_SAMPLES", "0"))
    if max_samples > 0:
        end_idx = min(end_idx, start_idx + max_samples)
    return start_idx, end_idx


# ─── E-VQA ───────────────────────────────────────────────────────────────────

class EVQADataset(torch.utils.data.Dataset):
    """
    Encyclopedic-VQA test set.

    Requires:
      args.query_path  : path to test.csv
      args.image_dir   : directory where iNaturalist images live
                         (files named {dataset_image_ids}.jpg or .jpeg or .png)
                         Leave empty to attempt on-the-fly download from iNaturalist.
    """

    def __init__(self, args):
        self.args = args
        self.image_dir = getattr(args, "image_dir", None)
        self.data: List[Dict] = []
        with open(args.query_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(dict(row))

    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

    def _get_image(self, image_id: str) -> Image.Image:
        if self.image_dir:
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(self.image_dir, image_id + ext)
                if os.path.exists(p):
                    return Image.open(p).convert("RGB")
        # Fallback: iNaturalist CDN
        url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{image_id}/medium.jpg"
        img = _download_image(url)
        if img is not None:
            return img
        print(f"[!] Could not load image {image_id}; using black placeholder.")
        return Image.new("RGB", (224, 224), (0, 0, 0))

    def __getitem__(self, idx: int) -> Dict:
        row = self.data[idx]
        image_id = row["dataset_image_ids"]
        image = self._get_image(image_id)
        answer_str = row["answer"]
        answers = [a.strip() for a in answer_str.split("|") if a.strip()]
        return {
            "data_id": f"evqa_{image_id}_{idx}",
            "unique_id": f"evqa_{image_id}_{idx}",
            "question": row["question"],
            "answer": answers[0] if answers else "",
            "answers": answers,
            "wikipedia_url": row["wikipedia_url"],
            "evidence_section_id": row.get("evidence_section_id", ""),
            "image_path": image_id,
            "image_query": image,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)


# ─── InfoSeek ────────────────────────────────────────────────────────────────

class InfoSeekDataset(torch.utils.data.Dataset):
    """
    InfoSeek test set (CSV format matching encyclopedic_vqa structure).

    Requires:
      args.query_path  : path to infoseek_test_filtered.csv
      args.image_dir   : directory with OVEN images named {dataset_image_ids}.jpg etc.
    """

    def __init__(self, args):
        self.args = args
        self.image_dir = getattr(args, "image_dir", None)
        self.data: List[Dict] = []
        with open(args.query_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(dict(row))

    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

    def _get_image(self, image_id: str) -> Image.Image:
        if self.image_dir:
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(self.image_dir, image_id + ext)
                if os.path.exists(p):
                    return Image.open(p).convert("RGB")
        return Image.new("RGB", (224, 224), (0, 0, 0))

    def __getitem__(self, idx: int) -> Dict:
        row = self.data[idx]
        image_id = row.get("dataset_image_ids", str(idx))
        image = self._get_image(image_id)
        answer_str = row.get("answer", "")
        answers = [a.strip() for a in answer_str.split("|") if a.strip()]
        data_id = row.get("data_id", f"infoseek_{idx}")
        return {
            "data_id": data_id,
            "question": row["question"],
            "answer": answers[0] if answers else "",
            "answers": answers,
            "wikipedia_url": row.get("wikipedia_url", ""),
            "image_path": image_id,
            "image_query": image,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)


# ─── ViQuAE ──────────────────────────────────────────────────────────────────

class ViQuAEDataset(torch.utils.data.Dataset):
    """
    ViQuAE test set.

    Requires:
      args.query_path   : path to datasets/viquae_dataset/test.jsonl
      args.image_dir    : path to datasets/viquae_images/images/
      args.wiki_KB      : path to datasets/viquae_wikipedia/ (dir with .jsonl.gz files)
    """

    WIKI_FILES = ["humans_with_faces.jsonl.gz", "humans_without_faces.jsonl.gz", "non_humans.jsonl.gz"]

    def __init__(self, args):
        self.args = args
        self.image_dir = getattr(args, "image_dir", "")
        self.data: List[Dict] = []
        with open(args.query_path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        image_rel = sample["image"]
        image_path = os.path.join(self.image_dir, image_rel)
        image = _safe_image(image_path)
        answers_raw = sample.get("output", {}).get("answer", [])
        answer = answers_raw[0] if answers_raw else ""
        provenance = sample.get("output", {}).get("provenance", [])
        wiki_title = ""
        if provenance:
            titles = provenance[0].get("title", [])
            if titles:
                wiki_title = titles[0]
        return {
            "data_id": sample.get("id", str(idx)),
            "question": sample.get("original_question", sample.get("input", "")),
            "answer": answer,
            "answers": answers_raw,
            "wikipedia_url": sample.get("url", ""),
            "wiki_title": wiki_title,
            "image_path": image_path,
            "image_query": image,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)

    def load_wiki_kb(self) -> Dict:
        """Load ViQuAE Wikipedia KB from gzipped JSONL files. Returns dict keyed by title."""
        wiki_dir = getattr(self.args, "wiki_KB", "")
        kb: Dict[str, Any] = {}
        for fname in self.WIKI_FILES:
            fpath = os.path.join(wiki_dir, fname)
            if not os.path.exists(fpath):
                print(f"[ViQuAE] Warning: KB file not found: {fpath}")
                continue
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    title = entry.get("title", "")
                    if title:
                        kb[title] = entry
        print(f"[ViQuAE] Loaded {len(kb)} Wikipedia entries.")
        return kb


# ─── Vision-only datasets (no retrieval) ─────────────────────────────────────

class RealWorldQADataset(torch.utils.data.Dataset):
    """
    RealWorldQA — loads from local parquet files.

    Requires:
      args.query_path : glob pattern or directory for parquet files
                        e.g. datasets/_hf_raw/lmms-lab__RealWorldQA/data/test-*.parquet
    """

    def __init__(self, args):
        import glob
        from datasets import load_dataset

        self.args = args
        pattern = getattr(args, "query_path", "")
        files = sorted(glob.glob(pattern)) if "*" in pattern else [pattern]
        self.data = load_dataset("parquet", data_files=files, split="train")

    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.data[idx])
        img = sample.get("image")
        if img is not None and not isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif img is None:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return {
            "data_id": f"RealWorldQA_{idx}",
            "question": sample["question"],
            "answer": sample.get("answer", ""),
            "answers": [sample.get("answer", "")],
            "image_path": sample.get("image_path", ""),
            "image_query": img,
            "wikipedia_url": None,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)


class OCRBenchDataset(torch.utils.data.Dataset):
    """
    OCRBench — loads from local arrow/parquet.

    Requires:
      args.query_path : directory path e.g. datasets/ocrbench_test
    """

    def __init__(self, args):
        from datasets import load_from_disk
        self.args = args
        self.data = load_from_disk(getattr(args, "query_path", ""))

    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.data[idx])
        img = sample.get("image")
        if img is not None and not isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif img is None:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        answers = sample.get("answers", sample.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]
        return {
            "data_id": f"OCRBench_{idx}",
            "question": sample.get("question", ""),
            "answer": answers[0] if answers else "",
            "answers": answers,
            "category": sample.get("question_type", ""),
            "image_path": "",
            "image_query": img,
            "wikipedia_url": None,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)


class TextVQADataset(torch.utils.data.Dataset):
    """
    TextVQA validation — loads from local parquet files.

    Requires:
      args.query_path : glob/dir for lmms-lab__textvqa validation parquet files
    """

    def __init__(self, args):
        import glob
        from datasets import load_dataset

        self.args = args
        pattern = getattr(args, "query_path", "")
        files = sorted(glob.glob(pattern)) if "*" in pattern else [pattern]
        self.data = load_dataset("parquet", data_files=files, split="train")

    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.data[idx])
        img = sample.get("image")
        if img is not None and not isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif img is None:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        answers = sample.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        # Build question with OCR tokens (reference format)
        qs = sample.get("question", "")
        ocr_tokens = sample.get("ocr_tokens", [])
        if ocr_tokens:
            qs += "\nReference OCR tokens: " + ", ".join(ocr_tokens)
        qs += "\nAnswer the question using a single word or very few words. Answer: "
        return {
            "data_id": f"TextVQA_{sample.get('question_id', idx)}",
            "question": qs,
            "answer": answers[0] if answers else "",
            "answers": answers,
            "image_path": "",
            "image_query": img,
            "wikipedia_url": None,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)


class POPEDataset(torch.utils.data.Dataset):
    """
    POPE test — loads from local parquet files or arrow disk.

    Requires:
      args.query_path : glob pattern for parquet files or path to disk-saved dataset
                        e.g. datasets/_hf_raw/lmms-lab__POPE/data/test-*.parquet
    """

    def __init__(self, args):
        import glob
        from datasets import load_dataset, load_from_disk
        self.args = args
        pattern = getattr(args, "query_path", "")
        if os.path.isdir(pattern):
            parquet_files = sorted(glob.glob(os.path.join(pattern, "*.parquet")))
            if parquet_files:
                self.data = load_dataset("parquet", data_files=parquet_files, split="train")
                return
            # Fallback: arrow disk
            try:
                self.data = load_from_disk(pattern)
                return
            except Exception:
                pass
        # Glob pattern
        files = sorted(glob.glob(pattern))
        self.data = load_dataset("parquet", data_files=files, split="train")

    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.data[idx])
        img = sample.get("image")
        if img is not None and not isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif img is None:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        answers = sample.get("answer", sample.get("label", ""))
        if not isinstance(answers, list):
            answers = [str(answers)]
        return {
            "data_id": f"POPE_{sample.get('question_id', idx)}",
            "question": sample.get("question", ""),
            "answer": answers[0],
            "answers": answers,
            "category": sample.get("category", ""),
            "image_path": "",
            "image_query": img,
            "wikipedia_url": None,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)


class VStarDataset(torch.utils.data.Dataset):
    """
    V*Bench — loads from JSONL + local image files.

    Requires:
      args.query_path  : path to test_questions.jsonl
      args.image_dir   : root directory for images (e.g. datasets/vstar_bench_raw)
    """

    def __init__(self, args):
        self.args = args
        self.image_dir = getattr(args, "image_dir", "")
        self.data = []
        with open(args.query_path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        image_rel = sample["image"]
        image_path = os.path.join(self.image_dir, image_rel)
        image = _safe_image(image_path)
        answer = sample.get("label", "")
        return {
            "data_id": f"VStar_{sample.get('question_id', idx)}",
            "question": sample.get("text", ""),
            "answer": answer,
            "answers": [answer],
            "category": sample.get("category", ""),
            "image_path": image_path,
            "image_query": image,
            "wikipedia_url": None,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)


class ChartQADataset(torch.utils.data.Dataset):
    """
    ChartQA test — loads from local parquet files.

    Requires:
      args.query_path : glob/dir for lmms-lab__ChartQA test parquet files
    """

    def __init__(self, args):
        import glob
        from datasets import load_dataset
        self.args = args
        pattern = getattr(args, "query_path", "")
        files = sorted(glob.glob(pattern)) if "*" in pattern else sorted(glob.glob(os.path.join(pattern, "*.parquet")))
        self.data = load_dataset("parquet", data_files=files, split="train")

    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.data[idx])
        img = sample.get("image")
        if img is not None and not isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif img is None:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        answer = sample.get("answer", "")
        qs = sample.get("question", "") + "\nAnswer with a single word or very few words. Answer: "
        return {
            "data_id": f"ChartQA_{idx}",
            "question": qs,
            "answer": answer,
            "answers": [answer],
            "image_path": "",
            "image_query": img,
            "wikipedia_url": None,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)


class AMBERDataset(torch.utils.data.Dataset):
    """
    AMBER hallucination benchmark (generative queries).

    Requires:
      args.query_path  : path to query_all.json (or query_generative.json / query_discriminative.json)
      args.image_dir   : directory with AMBER images
    """

    def __init__(self, args):
        self.args = args
        self.image_dir = getattr(args, "image_dir", "")
        with open(args.query_path) as f:
            self.data = json.load(f)

    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        image_path = os.path.join(self.image_dir, sample["image"])
        image = _safe_image(image_path)
        return {
            "data_id": f"AMBER_{sample['id']}",
            "question": sample.get("query", ""),
            "answer": "",  # AMBER uses separate annotation file
            "answers": [],
            "image_path": image_path,
            "image_query": image,
            "wikipedia_url": None,
            "entity": None,
        }

    def __len__(self):
        return len(self.data)


# ─── Dispatcher ──────────────────────────────────────────────────────────────

_DATASET_MAP = {
    "evqa": EVQADataset,
    "infoseek": InfoSeekDataset,
    "viquae": ViQuAEDataset,
    "real_world_qa": RealWorldQADataset,
    "ocrbench": OCRBenchDataset,
    "textvqa": TextVQADataset,
    "pope": POPEDataset,
    "vstar": VStarDataset,
    "chartqa": ChartQADataset,
    "amber": AMBERDataset,
}


def get_dataset(args):
    """Return (dataset, start_idx, end_idx)."""
    name = args.dataset_name
    if name not in _DATASET_MAP:
        raise ValueError(f"Unknown dataset: {name!r}. Supported: {list(_DATASET_MAP)}")
    dataset = _DATASET_MAP[name](args)
    start_idx, end_idx = calculate_splits(len(dataset))
    dataset.split(start_idx, end_idx)
    return dataset, start_idx, end_idx
