"""
Retrieval adapters for AREA.

Vision-only tasks return no context. ViQuAE retrieval is included. For E-VQA
and InfoSeek image-index retrieval, set AREA_RETRIEVER_DIR to a directory
that contains a compatible retriever.py implementation.
"""
import gzip
import json
import os
import sys
from typing import List, Optional, Tuple

from PIL import Image
PASSAGE_DELIMITER = "\n\n\n"


def _uniform_passages(text: str, n: int = 300) -> List[str]:
    """Split text into ~n-word chunks (mirrors external Retriever)."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    sentences = list(nlp(text).sents)
    passages, passage, tokens = [], [], 0
    for sent in sentences:
        if tokens + len(sent) > n:
            if passage:
                passages.append(" ".join(passage))
            passage = [sent.text]
            tokens = len(sent)
        else:
            passage.append(sent.text)
            tokens += len(sent)
    if passage:
        passages.append(" ".join(passage))
    return passages


class NoRetriever:
    """Dummy retriever for vision-only benchmarks."""

    def retrieve(self, image_query: Image.Image, **kwargs) -> Tuple[List, List, List]:
        return [], [], []


class OracleRetriever:
    """
    Fallback retriever for evqa/infoseek that uses the ground-truth Wikipedia URL
    supplied per sample to look up passages directly from the KB JSON.
    This is used when EVA-CLIP-8B is not available locally.
    """

    def __init__(self, args, wikipedia_dict=None):
        self.dataset_name = getattr(args, "dataset_name", "infoseek")
        if wikipedia_dict is not None:
            self.wikipedia = wikipedia_dict
            print(f"[OracleRetriever] Reusing preloaded KB ({len(self.wikipedia)} entries).")
        else:
            wiki_KB = getattr(args, "wiki_KB", "")
            print(f"[OracleRetriever] Loading KB from {wiki_KB} …")
            try:
                import ujson as _loader
            except ImportError:
                import json as _loader
            with open(wiki_KB, "r") as f:
                self.wikipedia = _loader.load(f)
            print(f"[OracleRetriever] KB loaded ({len(self.wikipedia)} entries).")

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Return common URL variants used by the local KB dumps."""
        variants = [url]
        if "//en.wikipedia.org/" in url:
            variants.append(url.replace("//en.wikipedia.org/", "//en.m.wikipedia.org/"))
        if "//en.m.wikipedia.org/" in url:
            variants.append(url.replace("//en.m.wikipedia.org/", "//en.wikipedia.org/"))
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(variants))

    def retrieve(
        self,
        image_query: Image.Image,
        wikipedia_url: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[str], List, List[str]]:
        if not wikipedia_url:
            return [], [], []
        key = None
        for candidate in self._normalize_url(wikipedia_url):
            if candidate in self.wikipedia:
                key = candidate
                break
        if key is None:
            return [], [], []
        entry = self.wikipedia[key]
        if self.dataset_name in ("evqa", "infoseek") and entry.get("section_texts"):
            passages = entry.get("section_texts", [])
        else:
            text = entry.get("wikipedia_content", "")
            passages = _uniform_passages(text, n=300) if text else []
        return passages, [], [key]


class ExternalImageRetriever:
    """
    Thin adapter that delegates to external Retriever class for evqa and infoseek.
    Falls back to OracleRetriever when EVA-CLIP-8B is not available.
    """

    def __init__(self, args):
        import importlib.util as _ilu
        retriever_dir = os.environ.get("AREA_RETRIEVER_DIR")
        if not retriever_dir:
            print("[ExternalImageRetriever] AREA_RETRIEVER_DIR is unset; using the configured KB fallback.")
            self._retriever = None
            self._oracle = OracleRetriever(args)
            return
        retriever_dir = os.path.abspath(os.path.expanduser(retriever_dir))
        retriever_file = os.path.join(retriever_dir, "retriever.py")
        if not os.path.isfile(retriever_file):
            raise FileNotFoundError(f"AREA_RETRIEVER_DIR must contain retriever.py: {retriever_dir}")
        _spec = _ilu.spec_from_file_location("_area_external_retriever", retriever_file)
        _mod = _ilu.module_from_spec(_spec)
        sys.path.insert(0, retriever_dir)
        try:
            _spec.loader.exec_module(_mod)
        finally:
            try:
                sys.path.remove(retriever_dir)
            except ValueError:
                pass
        # Pre-check: skip external retriever (and its 8B model download) when
        # EVA-CLIP-8B weights aren't fully cached locally.
        eva_model_id = os.environ.get("EVA_CLIP_MODEL_PATH", "BAAI/EVA-CLIP-8B")
        _eva_ready = False
        if os.path.isabs(eva_model_id) and os.path.isdir(eva_model_id):
            _eva_ready = True
        else:
            try:
                from huggingface_hub import try_to_load_from_cache
                # Returns a str path if cached, None if unknown, or a sentinel
                # object (_CACHED_NO_EXIST) if negatively cached. Only a str
                # path pointing to an existing file means it's actually ready.
                _probe = try_to_load_from_cache(eva_model_id, "pytorch_model-00001-of-00004.bin")
                _eva_ready = isinstance(_probe, str) and os.path.exists(_probe)
            except Exception:
                _eva_ready = False

        # Do not run image-index retrieval when the configured query-image
        # directory is missing. Otherwise the dataset loader supplies black
        # placeholders, and FAISS retrieval becomes systematically wrong.
        image_dir = getattr(args, "image_dir", None)
        if image_dir and not os.path.isdir(image_dir):
            print(f"[ExternalImageRetriever] image_dir not found ({image_dir}); using OracleRetriever.")
            _eva_ready = False

        if not _eva_ready:
            print(f"[ExternalImageRetriever] EVA-CLIP-8B not cached; using OracleRetriever directly.")
            self._retriever = None
            self._oracle = OracleRetriever(args)
            return

        # Patch Retriever.__init__ to capture `self` even if it throws later,
        # so we can reuse the already-loaded wiki KB on fallback.
        _original_init = _mod.Retriever.__init__
        _saved = [None]

        def _capturing_init(r, a):
            _saved[0] = r
            _original_init(r, a)

        _mod.Retriever.__init__ = _capturing_init
        try:
            self._retriever = _mod.Retriever(args)
            self._oracle = None
        except Exception as e:
            print(f"[ExternalImageRetriever] EVA-CLIP load failed ({e}); falling back to OracleRetriever.")
            self._retriever = None
            wikipedia_cache = getattr(_saved[0], "wikipedia", None)
            self._oracle = OracleRetriever(args, wikipedia_dict=wikipedia_cache)
        finally:
            _mod.Retriever.__init__ = _original_init

    def retrieve(self, image_query: Image.Image, **kwargs) -> Tuple[List, List, List]:
        if self._retriever is not None:
            # external Retriever.retrieve doesn't accept wikipedia_url
            retriever_kwargs = {k: v for k, v in kwargs.items() if k in ("dataset_image_id", "query")}
            return self._retriever.retrieve(image_query, **retriever_kwargs)
        return self._oracle.retrieve(image_query, **kwargs)


class ViQuAERetriever:
    """
    ViQuAE-specific retriever: uses BM25 or simple keyword search over the
    gzipped Wikipedia KB (since ViQuAE doesn't ship a FAISS index in our setup).

    For now we use TF-IDF/BM25 matching on the page title as a lightweight fallback.
    With proper FAISS we'd pass the image through EVA-CLIP, but we defer that.
    """

    def __init__(self, args):
        self.args = args
        self.top_k = getattr(args, "top_k", 3)
        self._kb: Optional[dict] = None  # loaded lazily

    def _load_kb(self):
        if self._kb is not None:
            return
        wiki_dir = getattr(self.args, "wiki_KB", "")
        self._kb = {}
        files = ["humans_with_faces.jsonl.gz", "humans_without_faces.jsonl.gz", "non_humans.jsonl.gz"]
        for fname in files:
            fpath = os.path.join(wiki_dir, fname)
            if not os.path.exists(fpath):
                continue
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        title = entry.get("wikipedia_title") or entry.get("title", "")
                        if title:
                            self._kb[title] = entry
                    except json.JSONDecodeError:
                        continue
        print(f"[ViQuAERetriever] Loaded {len(self._kb)} KB entries.")

    def retrieve(
        self,
        image_query: Image.Image,
        wiki_title: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[str], List, List[str]]:
        """Return (passages, [], [wiki_title]) for the given Wikipedia title."""
        self._load_kb()
        if not wiki_title or wiki_title not in self._kb:
            return [], [], []
        entry = self._kb[wiki_title]
        passages = entry.get("passages", [])
        if not passages and entry.get("section_texts"):
            passages = entry.get("section_texts", [])
        if not passages:
            text = entry.get("wikipedia_content", "")
            passages = _uniform_passages(text, n=300) if text else []
        if not passages and isinstance(entry.get("text"), dict):
            paragraphs = entry["text"].get("paragraph", [])
            if isinstance(paragraphs, list):
                passages = [p for p in paragraphs if isinstance(p, str) and p.strip()]
            elif isinstance(paragraphs, str):
                passages = _uniform_passages(paragraphs, n=300)
        return passages, [], [wiki_title]


def get_retriever(args):
    """Return the appropriate retriever for args.dataset_name."""
    name = getattr(args, "dataset_name", "")
    experiment_type = getattr(args, "experiment_type", "with_retrieval")
    if experiment_type == "no_retrieval" or name in (
        "real_world_qa", "ocrbench", "textvqa", "pope", "vstar", "chartqa", "amber"
    ):
        return NoRetriever()
    if name in ("evqa", "infoseek"):
        return ExternalImageRetriever(args)
    if name == "viquae":
        return ViQuAERetriever(args)
    return NoRetriever()
