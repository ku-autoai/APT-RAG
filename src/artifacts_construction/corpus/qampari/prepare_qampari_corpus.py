#!/usr/bin/env python3
"""Convert official QAMPARI chunked Wikipedia JSONL shards into corpus.jsonl.

Official shards use `{id, contents, meta}`. The indexer expects `document_id` and `text`.
This rewrite matches the field layout used to build the paper QAMPARI index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chunks import find_chunk_files, iter_converted


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "scripts").is_dir() and (parent / "src" / "artifacts_construction").is_dir():
            return parent
    raise RuntimeError("cannot locate APT-RAG repository root")


def build_arg_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory that contains wikipedia_chunks_*.jsonl (or chunks_v5/)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Write at most N records (debug)")
    parser.add_argument("--dry-run", action="store_true", help="Count shards/records without writing")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    root = args.repo_root.resolve()
    input_dir = (args.input_dir or (root / "data" / "raw" / "qampari" / "chunked_wikipedia")).resolve()
    output = (args.output or (root / "data" / "artifacts" / "corpus" / "qampari" / "corpus.jsonl")).resolve()
    files = find_chunk_files(input_dir)
    if not files:
        raise FileNotFoundError(
            f"no wikipedia_chunks_*.jsonl under {input_dir}. "
            "Extract chunked_wikipedia.tar.gz first."
        )
    print(f"[qampari-corpus] shards={len(files)} input={input_dir}")
    if args.dry_run:
        n = 0
        for _ in iter_converted(files):
            n += 1
            if args.limit is not None and n >= args.limit:
                break
        print(f"[qampari-corpus] dry-run records={n}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", encoding="utf-8") as f:
        for row in iter_converted(files):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.limit is not None and n >= args.limit:
                break
            if n % 500000 == 0:
                print(f"[qampari-corpus] wrote {n:,} records", flush=True)
    print(f"[qampari-corpus] wrote {n:,} records -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
