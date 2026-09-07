# APT-RAG: Adaptive Planning and Topology-aware Evidence Gathering for Evidence-Intensive QA (Findings of EMNLP 2026)

Official code for **A Tree-based RAG Framework for Evidence-Intensive QA via Adaptive Planning and Topology-Aware Evidence Gathering**, by Songeun Lee, Kyungjin Min, Injae Na, Suyeong Lee, Chiyoung Kim, Woohwan Jung.

[**📄 Paper**](https://arxiv.org/abs/2609.04981) | [**🤗 Artifacts**](https://huggingface.co/datasets/KJ-Min/APT-RAG-artifacts)

## Overview 

Evidence-intensive QA requires synthesizing information scattered across dozens or even hundreds of documents, exposing two fundamental limitations of existing structured RAG methods: **structural rigidity**, which prevents dynamic reasoning expansion, and **topology-ignorant evidence gathering**, which fails to integrate evidence across different reasoning nodes.
 
To address these limitations, we propose **APT-RAG**, an **A**daptive **P**lanning and **T**opology-aware evidence gathering RAG framework. APT-RAG recursively constructs a reasoning tree in a depth-first manner and resolves each node through the following four-step procedure:
 
- **Contextualization:** Resolves references to preceding sibling answers to produce a self-contained sub-question.
- **Adaptive Planning:** Determines the evidence gathering strategy—reusing sibling QA pairs (lateral), directly retrieving from the corpus (external), or decomposing into sub-questions (vertical).
- **Topology-aware Evidence Gathering:** Gathers evidence along the reasoning topology through lateral, external, or vertical strategies.
- **Answer Generation:** Generates a sub-answer that can subsequently serve as evidence for sibling or parent nodes.
APT-RAG further introduces **Evidence-Guided Batched Answer Generation**, which clusters sibling sub-questions sharing overlapping retrieved evidence and answers them jointly, substantially reducing LLM calls and inference latency.
 
<p align="center">
  <img src="figure1.png" width="80%" />
</p>
 


## Requirements

```
pip install -r requirements.txt
```

If you run an OpenAI-compatible evaluation model, set the API key through the environment:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Datasets

- In our paper, we use two evidence-intensive QA benchmarks that require synthesizing information distributed across dozens to hundreds of documents.
- [**MoNaCo**](https://arxiv.org/pdf/2508.11133): A benchmark of complex information-seeking questions requiring synthesis across 43.3 documents on average. 
- [**QAMPARI**](https://arxiv.org/pdf/2205.12665): A multi-answer open-domain QA benchmark where answers are distributed across multiple passages, requiring aggregation over 13.0 documents on average. 

Benchmark files, retrieval corpora, vector indices, and generated outputs are not included in this repository because they may contain benchmark queries, gold answers, retrieved passages, or large artifacts. Download the official sources into `data/raw/`, then convert them with the scripts below.

### Download official sources

**MoNaCo** is gated on Hugging Face ([`allenai/MoNaCo_Benchmark`](https://huggingface.co/datasets/allenai/MoNaCo_Benchmark)). Accept the license, then:

```bash
hf download allenai/MoNaCo_Benchmark --repo-type dataset --local-dir data/raw/monaco
```

After the download, you should have `data/raw/monaco/monaco_version_1_release.jsonl` and `data/raw/monaco/execution_traces/`.

**QAMPARI** files come from the [official website](https://samsam3232.github.io/qampari/):

- Question zip (`https://aggreg-qa.s3.amazonaws.com/qampari.zip`) → unzip under `data/raw/qampari/` so that `data/raw/qampari/qampari/test_data.jsonl` exists.
- Chunked Wikipedia Dump (`https://aggreg-qa.s3.amazonaws.com/chunked_wikipedia.tar.gz`) → extract under `data/raw/qampari/chunked_wikipedia/`.

### Convert to the repo layout

QAMPARI gold matching needs the chunked Wikipedia dump (or an already converted `corpus.jsonl`).

```bash
bash scripts/prepare_benchmarks.sh
```

This writes:

```
data/benchmarks/monaco/monaco_full.jsonl
data/benchmarks/monaco/monaco_300.jsonl
data/benchmarks/monaco/execution_traces/
data/benchmarks/qampari/qampari_full.jsonl
data/benchmarks/qampari/qampari_300.jsonl
```

The 300-example subsets are filtered by the committed ID lists in `data/prep/`. Use `--skip-monaco` or `--skip-qampari` to convert a single dataset.

## Corpus and Index Construction

Retrieval corpora and vector indices live under `data/artifacts/`. All downstream scripts read from these fixed paths:

| Dataset  | Corpus path | Index path |
|----------|-------------|------------|
| MoNaCo   | `data/artifacts/corpus/monaco/corpus_1M.jsonl` | `data/artifacts/vectorDB/monaco/` |
| QAMPARI  | `data/artifacts/corpus/qampari/corpus.jsonl` | `data/artifacts/vectorDB/qampari/` |

### Released artifacts (recommended)

The MoNaCo corpus and both FAISS HNSW indices (`Qwen/Qwen3-Embedding-0.6B`) are released at [`KJ-Min/APT-RAG-artifacts`](https://huggingface.co/datasets/KJ-Min/APT-RAG-artifacts) (CC BY-SA 4.0; Wikipedia-derived). The QAMPARI passage corpus is not included there; build it with `prepare_qampari_corpus.sh` below.

```bash
hf download KJ-Min/APT-RAG-artifacts --repo-type dataset --local-dir data/artifacts
gzip -dc data/artifacts/corpus/monaco/corpus_1M.jsonl.gz \
  > data/artifacts/corpus/monaco/corpus_1M.jsonl
```

Expected layout after download and decompress:

```
data/artifacts/
  corpus/monaco/corpus_1M.jsonl
  vectorDB/monaco/index.pkl
  vectorDB/monaco/hnsw_sq8/hnsw.index
  vectorDB/qampari/index.pkl
  vectorDB/qampari/hnsw_sq8/hnsw.index
```

QAMPARI's index is large; the retrieval server may take a long time to load `index.pkl` before it starts serving.

### Rebuilding the MoNaCo corpus

Scripts: `src/artifacts_construction/corpus/monaco/`

1. **Parse** — `wikidump_parser.py` converts a Wikipedia `.xml.bz2` dump into JSONL.
2. **Chunk** — `corpus_chunker.py` splits passages into chunks (default: 512 tiktoken tokens).
3. **Sample** — `corpus_sampler.py` retains all gold passages (from MoNaCo execution traces) and samples 1M non-gold document groups.

Gold provenance traces must be placed at:
`data/benchmarks/monaco/execution_traces/dataset_ex_*.json`

```bash
# Step 1: parse dump (see wikidump_parser.py --help for multi-worker usage)
python src/artifacts_construction/corpus/monaco/wikidump_parser.py \
  --input /path/to/enwiki.xml.bz2 \
  --output-dir /tmp/monaco_corpus 0 1 --merge

# Step 2: chunk
python src/artifacts_construction/corpus/monaco/corpus_chunker.py \
  --input /tmp/monaco_corpus/corpus.jsonl \
  --output /tmp/monaco_corpus/corpus_chunked.jsonl

# Step 3: sample → final corpus
python src/artifacts_construction/corpus/monaco/corpus_sampler.py \
  --corpus /tmp/monaco_corpus/corpus_chunked.jsonl \
  --trace-dir data/benchmarks/monaco/execution_traces \
  --output data/artifacts/corpus/monaco/corpus_1M.jsonl
```

### Rebuilding the QAMPARI corpus

Convert the official chunked Wikipedia dump (do not place the raw `{id, contents, meta}` shards at `corpus.jsonl`; the indexer expects the converted schema):

```bash
bash scripts/prepare_qampari_corpus.sh
```

This writes `data/artifacts/corpus/qampari/corpus.jsonl` (large; the conversion can take a while). This file is required for QAMPARI inference even if you downloaded the released index.

### Vector Index

If you downloaded the released indices above, skip this step. To rebuild from a local corpus, both datasets use `Qwen/Qwen3-Embedding-0.6B` and write `hnsw_sq8/hnsw.index` + `index.pkl`. Index scripts launch `src/artifacts_construction/vectorDB/indexer/corpus_indexer.py` via `systemd-run` (requires a Linux user systemd session).

```bash
# MoNaCo
bash scripts/run_index_monaco.sh <gpu_id>

# QAMPARI
bash scripts/run_index_qampari.sh <gpu_id>
```

## Retrieval Server

After the index is built, start a FastAPI retrieval server. Baseline scripts query these servers at runtime.

| Dataset  | Server script | Default port |
|----------|---------------|--------------|
| MoNaCo   | `scripts/run_server_monaco.sh` | `8007` |
| QAMPARI  | `scripts/run_server_qampari.sh` | `8008` |

```bash
# MoNaCo
bash scripts/run_server_monaco.sh <gpu_id>

# QAMPARI
bash scripts/run_server_qampari.sh <gpu_id>
```

Use a different GPU from inference. Wait until the server finishes loading the index. To stop a server:

```bash
systemctl --user stop vectorstore-monaco.scope
systemctl --user stop vectorstore-qampari.scope
```

## APT-RAG

Entry point: `src/baselines/apt-rag/main.py`

Before running APT-RAG, start the retrieval server for your dataset (see [Retrieval Server](#retrieval-server)) and ensure the benchmark queries and corpus are available under `data/`.

```bash
# MoNaCo + QAMPARI, both LLM sizes
METHOD_FILTER=apt-rag bash scripts/gen_all_baselines_ans.sh

# Single dataset / model
DATASET_FILTER=monaco MODEL_FILTER=Qwen3-30B-A3B-Instruct-2507 METHOD_FILTER=apt-rag \
  bash scripts/gen_all_baselines_ans.sh

# 300-example subset
SAMPLE=sample_300 METHOD_FILTER=apt-rag bash scripts/gen_all_baselines_ans.sh
```

Results are saved under `results/{dataset}/apt-rag/answer/{model_name}/`. Each example writes `ex_{id}_answer.json`, `ex_{id}_retrieved.json`, and `execution_traces/ex_{id}.json`.

To shard across GPUs, set `APT_RAG_NUM_SHARDS` and `APT_RAG_SHARD_INDEX` on the same script. Each worker skips examples whose trace file already exists.

## Baselines

The repository includes the following inference baselines under `src/baselines/`:

| Method | Entry point |
|--------|-------------|
| LLM-only | `src/baselines/llm_only/LLM_only.py` |
| Naive RAG | `src/baselines/naive-rag/LLM_DPR.py` |
| Plan-RAG | `src/baselines/plan-rag/main.py` |
| Tree-of-Question | `src/baselines/tree-of-question/main.py` |
| RT-RAG | `src/baselines/rt-rag/main.py` |
| Logic-RAG | `src/baselines/logic-rag/main.py` |
| APT-RAG | `src/baselines/apt-rag/main.py` |

Run all methods through the unified script:

```bash
bash scripts/gen_all_baselines_ans.sh
```

Useful filters:

```bash
DATASET_FILTER=monaco METHOD_FILTER=apt-rag bash scripts/gen_all_baselines_ans.sh
MODEL_FILTER=Qwen3-4B-Instruct-2507 METHOD_FILTER=naive-rag bash scripts/gen_all_baselines_ans.sh
DRY_RUN=1 bash scripts/gen_all_baselines_ans.sh
```

## Evaluation

Answer, retrieval, and cost evaluation scripts are grouped under `src/evaluation/` and wrapped by shell scripts in `scripts/`.

```bash
# Answer quality
bash scripts/eval_all_baselines_ans.sh

# Retrieval metrics
bash scripts/eval_all_baselines_ret.sh

# Cost and trace statistics
bash scripts/eval_all_baselines_cost.sh
```

The scripts skip missing inputs and can be filtered with `DATASET_FILTER`, `MODEL_FILTER`, and `METHOD_FILTER` in the same way as generation. Use `DRY_RUN=1` to print commands without writing new outputs.

Generated answers, execution traces, detailed LLM judge calls, retrieval metrics, and cost metrics are written under `results/`. These files are intentionally excluded from version control because they may contain benchmark questions, gold answers, retrieved evidence, and model outputs.

## Repository Contents

This code release is intended to include:

- Source code for APT-RAG and baseline inference under `src/baselines/`.
- Evaluation code under `src/evaluation/`.
- Corpus/index construction and retrieval server code under `src/artifacts_construction/`.
- Benchmark conversion scripts and the paper 300-subset ID lists under `data/prep/`.
- Shell scripts under `scripts/`.
- Dependency metadata such as `requirements.txt`.

This repository is not intended to include:

- Official downloads under `data/raw/`.
- Benchmark question files or gold annotations under `data/benchmarks/`.
- Retrieval corpora under `data/artifacts/corpus/`.
- Vector database files under `data/artifacts/vectorDB/`.
- Generated answers, execution traces, evaluation logs, or metric outputs under `results/`.
- Local credentials, API keys, cache files, or model checkpoints.

## Citation

```bibtex
@inproceedings{lee-etal-2026-apt-rag,
  title     = {A Tree-based {RAG} Framework for Evidence-Intensive {QA} via Adaptive Planning and Topology-Aware Evidence Gathering},
  author    = {Lee, Songeun and Min, Kyungjin and Na, Injae and Lee, Suyeong and Kim, Chiyoung and Jung, Woohwan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  note      = {To appear}
}
```