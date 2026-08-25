from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CHUNK_NAME_RE = re.compile(r"wikipedia_chunks_(\d+)\.jsonl$")


def chunk_sort_key(path: Path) -> tuple:
    match = CHUNK_NAME_RE.search(path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def find_chunk_files(raw_dir: Path) -> List[Path]:
    candidates = [
        raw_dir / "chunks_v5",
        raw_dir,
        raw_dir / "chunked_wikipedia" / "chunks_v5",
        raw_dir / "chunked_wikipedia",
        raw_dir / "wikipedia_chunks" / "chunks_v5",
        raw_dir / "wikipedia_chunks",
    ]
    seen = set()
    files: List[Path] = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        for path in directory.glob("wikipedia_chunks_*.jsonl"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
        if files:
            break
    if not files:
        for path in raw_dir.rglob("wikipedia_chunks_*.jsonl"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    files.sort(key=chunk_sort_key)
    return files


def as_int(value: Any, default: int = -1) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def chunk_index_from_id(raw_id: str) -> int:
    if "__" not in raw_id:
        return 0
    tail = raw_id.rsplit("__", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


def convert_chunk(row: Dict[str, Any]) -> Dict[str, Any] | None:
    raw_id = str(row.get("id") or "").strip()
    text = row.get("contents") or row.get("text") or ""
    if not raw_id or not text:
        return None
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    chunk_index = chunk_index_from_id(raw_id)
    page_id = as_int(meta.get("page_id"), default=-1)
    return {
        "document_id": f"{raw_id}:0",
        "text": text,
        "title": meta.get("title") or "",
        "page_id": page_id,
        "url": meta.get("url") or "",
        "revid": "" if meta.get("revid") is None else str(meta.get("revid")),
        "language": "en",
        "section": "",
        "section_level": -1,
        "content_type": "sentence",
        "chunk_index": chunk_index,
        "ns": "",
    }


def iter_converted(files: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] skip {path}:{line_num}: {exc}", file=sys.stderr)
                    continue
                converted = convert_chunk(row)
                if converted is None:
                    print(f"[WARN] skip {path}:{line_num}: missing id/contents", file=sys.stderr)
                    continue
                yield converted
