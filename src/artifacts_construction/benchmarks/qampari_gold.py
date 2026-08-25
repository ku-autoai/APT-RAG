"""Match QAMPARI proof annotations to official chunked-Wikipedia passages.

Index chunks by page title, then for each proof attach the matching corpus
passage as `{title, content}`. Only pages cited by `found_in_url` are kept
in memory.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set
from urllib.parse import unquote


MIN_OVERLAP = 0.65


def _load_chunks_module():
    path = Path(__file__).resolve().parents[1] / "corpus" / "qampari" / "chunks.py"
    spec = importlib.util.spec_from_file_location("qampari_chunks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chunks = _load_chunks_module()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def url_to_title(url: str) -> str:
    if not url:
        return ""
    title = url.rstrip("/").split("/")[-1]
    title = unquote(title)
    title = title.replace("_", " ")
    return title


def dedup_gold_docs(gold_doc: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for doc in gold_doc:
        key = (doc["title"], doc["content"])
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped


def iter_proofs(example: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    answers = example.get("validated_answer") or example.get("answer_list") or []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        for proof in ans.get("proof") or []:
            if not isinstance(proof, dict):
                continue
            yield url_to_title(proof.get("found_in_url") or ""), proof.get("proof_text") or ""


def collect_needed_titles(examples: Sequence[Mapping[str, Any]]) -> Set[str]:
    titles = set()
    for example in examples:
        for title, _proof_text in iter_proofs(example):
            if title:
                titles.add(title)
    return titles


def token_overlap_score(proof_text: str, chunk_text: str) -> float:
    proof_tokens = normalize_text(proof_text).split()
    chunk_tokens = set(normalize_text(chunk_text).split())
    if not proof_tokens:
        return 0.0
    matched = sum(1 for tok in proof_tokens if tok in chunk_tokens)
    return matched / len(proof_tokens)


def find_matching_chunks(
    title: str,
    proof_text: str,
    title_to_chunks: dict[str, list[dict]],
    min_overlap: float = MIN_OVERLAP,
) -> list[dict]:
    candidates = title_to_chunks.get(title, [])
    if not candidates or not proof_text:
        return []

    norm_proof = normalize_text(proof_text)

    exact_matches = []
    for chunk in candidates:
        if norm_proof in normalize_text(chunk["content"]):
            exact_matches.append(chunk)
    if exact_matches:
        return exact_matches

    contained_matches = []
    for chunk in candidates:
        norm_chunk = normalize_text(chunk["content"])
        if norm_chunk and norm_chunk in norm_proof:
            contained_matches.append(chunk)
    if contained_matches:
        return contained_matches

    scored = []
    for chunk in candidates:
        score = token_overlap_score(proof_text, chunk["content"])
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] >= min_overlap:
        return [scored[0][1]]
    return []


def match_gold_docs(
    example: Mapping[str, Any],
    title_to_chunks: dict[str, list[dict]],
) -> tuple[list[dict], int]:
    gold_doc = []
    n_missing = 0
    for title, proof_text in iter_proofs(example):
        matched = find_matching_chunks(title, proof_text, title_to_chunks)
        if matched:
            for chunk in matched:
                gold_doc.append({"title": chunk["title"], "content": chunk["content"]})
        else:
            n_missing += 1
    return dedup_gold_docs(gold_doc), n_missing


def _append_if_needed(
    title_to_chunks: dict[str, list[dict]],
    needed_titles: Set[str],
    title: str,
    text: str,
) -> bool:
    if not title or not text or title not in needed_titles:
        return False
    title_to_chunks.setdefault(title, []).append({"title": title, "content": text})
    return True


def load_title_to_chunks(
    needed_titles: Set[str],
    corpus_jsonl: Optional[Path] = None,
    raw_chunks_dir: Optional[Path] = None,
) -> dict[str, list[dict]]:
    if not needed_titles:
        return {}

    title_to_chunks: dict[str, list[dict]] = {}
    kept = 0
    scanned = 0

    if corpus_jsonl is not None and corpus_jsonl.is_file():
        print(f"[qampari-gold] indexing {len(needed_titles)} titles from {corpus_jsonl}")
        with corpus_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                scanned += 1
                if not line.strip():
                    continue
                item = json.loads(line)
                if _append_if_needed(
                    title_to_chunks,
                    needed_titles,
                    item.get("title") or "",
                    item.get("text") or "",
                ):
                    kept += 1
                if scanned % 500000 == 0:
                    print(f"[qampari-gold] scanned {scanned:,} corpus rows, kept {kept:,}", flush=True)
    else:
        if raw_chunks_dir is None:
            raise FileNotFoundError("QAMPARI gold matching needs --qampari-corpus or raw chunked Wikipedia")
        files = _chunks.find_chunk_files(raw_chunks_dir)
        if not files:
            raise FileNotFoundError(
                f"no wikipedia_chunks_*.jsonl under {raw_chunks_dir}. "
                "Extract chunked_wikipedia.tar.gz first."
            )
        print(f"[qampari-gold] indexing {len(needed_titles)} titles from {len(files)} raw shards")
        for item in _chunks.iter_converted(files):
            scanned += 1
            if _append_if_needed(title_to_chunks, needed_titles, item.get("title") or "", item.get("text") or ""):
                kept += 1
            if scanned % 500000 == 0:
                print(f"[qampari-gold] scanned {scanned:,} raw chunks, kept {kept:,}", flush=True)

    print(
        f"[qampari-gold] scanned={scanned:,} kept_chunks={kept:,} "
        f"titles_with_chunks={sum(1 for v in title_to_chunks.values() if v)}/"
        f"{len(needed_titles)}"
    )
    return title_to_chunks
