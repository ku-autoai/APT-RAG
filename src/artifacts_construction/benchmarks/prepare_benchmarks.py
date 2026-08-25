#!/usr/bin/env python3
"""Convert official MoNaCo / QAMPARI downloads into the JSONL layout this repo expects.

Outputs (under data/benchmarks/ unless overridden):
  monaco/monaco_full.jsonl
  monaco/monaco_300.jsonl
  monaco/execution_traces/dataset_ex_*.json
  qampari/qampari_full.jsonl
  qampari/qampari_300.jsonl

QAMPARI retrieval gold (`gold_doc`) is computed from official proof annotations
matched against chunked Wikipedia. 300-subsets are filtered by the committed
ex_num lists in data/prep/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from qampari_gold import collect_needed_titles, load_title_to_chunks, match_gold_docs


QID_TYPE_MAP = {
    "wikidata_simple": ("simple", "sentence"),
    "wikidata_comp": ("composition", "sentence"),
    "wikidata_intersection": ("intersection", "sentence"),
    "wikitables_simple": ("simple", "table"),
    "wikitables_composition": ("composition", "table"),
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "scripts").is_dir() and (parent / "src" / "artifacts_construction").is_dir():
            return parent
    raise RuntimeError("cannot locate APT-RAG repository root")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_num}: {exc}") from exc


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_id_list(path: Path) -> List[int]:
    ids: List[int] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ex_num = int(line)
            except ValueError as exc:
                raise ValueError(f"invalid ex_num in {path}:{line_num}: {line!r}") from exc
            if ex_num in seen:
                raise ValueError(f"duplicate ex_num {ex_num} in {path}")
            seen.add(ex_num)
            ids.append(ex_num)
    if not ids:
        raise ValueError(f"no ex_num values found in {path}")
    return ids


def rows_by_ex_num(rows: Iterable[Mapping[str, Any]], source: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        if "ex_num" not in row:
            raise ValueError(f"{source} row {idx} is missing ex_num")
        ex_num = int(row["ex_num"])
        if ex_num in out:
            raise ValueError(f"duplicate ex_num {ex_num} in {source}")
        out[ex_num] = dict(row)
    return out


def filter_by_ids(
    by_ex: Mapping[int, Mapping[str, Any]],
    ids: Sequence[int],
    source_name: str,
) -> List[Dict[str, Any]]:
    missing = [ex_num for ex_num in ids if ex_num not in by_ex]
    if missing:
        preview = ", ".join(str(x) for x in missing[:10])
        more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        raise FileNotFoundError(
            f"{len(missing)} {source_name} ex_num values are missing from the converted full set: {preview}{more}"
        )
    return [dict(by_ex[ex_num]) for ex_num in ids]


def first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.is_file():
            return path
    return None


def first_existing_dir(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.is_dir():
            return path
    return None


def copy_monaco_full(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    n = 0
    for _ in iter_jsonl(dst):
        n += 1
    return n


def copy_monaco_traces(src_dir: Path, dst_dir: Path) -> int:
    files = sorted(src_dir.glob("dataset_ex_*.json"))
    if not files:
        raise FileNotFoundError(f"no dataset_ex_*.json traces under {src_dir}")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dst_dir / src.name)
    return len(files)


def parse_qid(qid: str) -> tuple[int, str, str]:
    parts = str(qid).split("__")
    if len(parts) < 2:
        raise ValueError(f"unexpected QAMPARI qid: {qid!r}")
    try:
        ex_num = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"QAMPARI qid does not start with an integer: {qid!r}") from exc
    mapped = QID_TYPE_MAP.get(parts[1])
    if mapped is None:
        raise ValueError(f"unknown QAMPARI qid type token {parts[1]!r} in {qid!r}")
    return ex_num, mapped[0], mapped[1]


def convert_qampari_row(
    row: Mapping[str, Any],
    row_id: int,
    gold_doc: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if "qid" not in row:
        raise ValueError(f"QAMPARI row {row_id} is missing qid")
    ex_num, question_type, content_type = parse_qid(row["qid"])
    question = row.get("question_text", row.get("question"))
    if not question:
        raise ValueError(f"QAMPARI row {row_id} is missing question_text")
    validated = row.get("answer_list", row.get("validated_answer"))
    if validated is None:
        raise ValueError(f"QAMPARI row {row_id} is missing answer_list")
    return {
        "id": row_id,
        "ex_num": ex_num,
        "question": question,
        "question_type": question_type,
        "content_type": content_type,
        "validated_answer": validated,
        "gold_doc": list(gold_doc),
    }


def convert_qampari(
    examples: Sequence[Mapping[str, Any]],
    title_to_chunks: Mapping[str, List[Dict[str, Any]]],
    source: Path,
) -> tuple[List[Dict[str, Any]], int]:
    rows = []
    seen_ex = set()
    n_missing_proofs = 0
    for row_id, row in enumerate(examples):
        gold_doc, n_missing = match_gold_docs(row, title_to_chunks)
        n_missing_proofs += n_missing
        converted = convert_qampari_row(row, row_id, gold_doc)
        ex_num = converted["ex_num"]
        if ex_num in seen_ex:
            raise ValueError(f"duplicate QAMPARI ex_num {ex_num} in {source}")
        seen_ex.add(ex_num)
        rows.append(converted)
    if not rows:
        raise ValueError(f"no QAMPARI rows in {source}")
    return rows, n_missing_proofs


def build_arg_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--monaco-jsonl", type=Path, default=None)
    parser.add_argument("--monaco-traces", type=Path, default=None)
    parser.add_argument("--qampari-jsonl", type=Path, default=None)
    parser.add_argument("--qampari-corpus", type=Path, default=None,
                        help="Converted QAMPARI corpus.jsonl. If omitted, raw chunked Wikipedia is scanned.")
    parser.add_argument("--qampari-chunks-dir", type=Path, default=None)
    parser.add_argument("--monaco-300-ids", type=Path, default=None)
    parser.add_argument("--qampari-300-ids", type=Path, default=None)
    parser.add_argument("--benchmarks-dir", type=Path, default=None)
    parser.add_argument("--allow-missing-gold", action="store_true",
                        help="Write empty gold_doc instead of failing when Wikipedia chunks are missing.")
    parser.add_argument("--skip-monaco", action="store_true")
    parser.add_argument("--skip-qampari", action="store_true")
    return parser


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = args.repo_root.resolve()
    raw = root / "data" / "raw"
    prep = root / "data" / "prep"
    args.benchmarks_dir = (args.benchmarks_dir or (root / "data" / "benchmarks")).resolve()
    args.monaco_jsonl = args.monaco_jsonl or first_existing(
        [
            raw / "monaco" / "monaco_version_1_release.jsonl",
            raw / "monaco" / "monaco_full.jsonl",
        ]
    )
    args.monaco_traces = args.monaco_traces or first_existing_dir(
        [
            raw / "monaco" / "execution_traces",
        ]
    )
    args.qampari_jsonl = args.qampari_jsonl or first_existing(
        [
            raw / "qampari" / "qampari" / "test_data.jsonl",
            raw / "qampari" / "qampari_data" / "test_data.jsonl",
            raw / "qampari" / "test_data.jsonl",
        ]
    )
    args.qampari_corpus = args.qampari_corpus or first_existing(
        [
            root / "data" / "artifacts" / "corpus" / "qampari" / "corpus.jsonl",
        ]
    )
    args.qampari_chunks_dir = args.qampari_chunks_dir or first_existing_dir(
        [
            raw / "qampari" / "chunked_wikipedia",
            raw / "qampari" / "wikipedia_chunks",
        ]
    )
    args.monaco_300_ids = (args.monaco_300_ids or (prep / "monaco_300_ids.txt")).resolve()
    args.qampari_300_ids = (args.qampari_300_ids or (prep / "qampari_300_ids.txt")).resolve()
    return args


def prepare_monaco(args: argparse.Namespace) -> None:
    if args.monaco_jsonl is None:
        raise FileNotFoundError(
            "MoNaCo JSONL not found. Place monaco_version_1_release.jsonl under data/raw/monaco/ "
            "or pass --monaco-jsonl."
        )
    if args.monaco_traces is None:
        raise FileNotFoundError(
            "MoNaCo execution_traces/ not found under data/raw/monaco/. Pass --monaco-traces."
        )
    monaco_jsonl = Path(args.monaco_jsonl).resolve()
    traces_src = Path(args.monaco_traces).resolve()
    out_dir = args.benchmarks_dir / "monaco"
    full_path = out_dir / "monaco_full.jsonl"
    sample_path = out_dir / "monaco_300.jsonl"
    traces_dst = out_dir / "execution_traces"

    n_full = copy_monaco_full(monaco_jsonl, full_path)
    n_traces = copy_monaco_traces(traces_src, traces_dst)
    monaco_300 = filter_by_ids(
        rows_by_ex_num(iter_jsonl(full_path), full_path),
        load_id_list(args.monaco_300_ids),
        "MoNaCo",
    )
    write_jsonl(sample_path, monaco_300)
    print(f"[monaco] full={n_full} traces={n_traces} sample_300={len(monaco_300)}")
    print(f"[monaco] wrote {full_path}")
    print(f"[monaco] wrote {sample_path}")
    print(f"[monaco] wrote {traces_dst}")


def prepare_qampari(args: argparse.Namespace) -> None:
    if args.qampari_jsonl is None:
        raise FileNotFoundError(
            "QAMPARI test_data.jsonl not found. Unzip qampari.zip under data/raw/qampari/ "
            "or pass --qampari-jsonl."
        )
    src = Path(args.qampari_jsonl).resolve()
    examples = list(iter_jsonl(src))
    needed_titles = collect_needed_titles(examples)
    corpus_jsonl = Path(args.qampari_corpus).resolve() if args.qampari_corpus else None
    chunks_dir = Path(args.qampari_chunks_dir).resolve() if args.qampari_chunks_dir else None
    if not (corpus_jsonl and corpus_jsonl.is_file()) and chunks_dir is None:
        if args.allow_missing_gold:
            title_to_chunks = {}
            print("[qampari] no corpus/chunks found; writing empty gold_doc")
        else:
            raise FileNotFoundError(
                "QAMPARI gold matching needs converted corpus.jsonl or raw chunked Wikipedia. "
                "Pass --qampari-corpus or extract chunked_wikipedia.tar.gz under data/raw/qampari/."
            )
    else:
        title_to_chunks = load_title_to_chunks(
            needed_titles,
            corpus_jsonl=corpus_jsonl if corpus_jsonl and corpus_jsonl.is_file() else None,
            raw_chunks_dir=None if (corpus_jsonl and corpus_jsonl.is_file()) else chunks_dir,
        )
    converted, n_missing_proofs = convert_qampari(examples, title_to_chunks, src)
    empty_gold = sum(1 for row in converted if not row["gold_doc"])
    out_dir = args.benchmarks_dir / "qampari"
    full_path = out_dir / "qampari_full.jsonl"
    sample_path = out_dir / "qampari_300.jsonl"
    write_jsonl(full_path, converted)
    qampari_300 = filter_by_ids(
        rows_by_ex_num(converted, full_path),
        load_id_list(args.qampari_300_ids),
        "QAMPARI",
    )
    write_jsonl(sample_path, qampari_300)
    print(
        f"[qampari] full={len(converted)} sample_300={len(qampari_300)} "
        f"empty_gold={empty_gold} unmatched_proofs={n_missing_proofs}"
    )
    print(f"[qampari] wrote {full_path}")
    print(f"[qampari] wrote {sample_path}")


def main() -> int:
    args = resolve_paths(build_arg_parser().parse_args())
    args.benchmarks_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_monaco:
        prepare_monaco(args)
    if not args.skip_qampari:
        prepare_qampari(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
