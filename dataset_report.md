# AURORA — Comprehensive Dataset Research, Comparison, and Selection Report

- **Document status**: Version 1.0
- **Primary domain**: Code data pipeline, dataset curation and governance
- **Author**: Sahana (Hardly Humans AI Club, IIT Dharwad)
- **Target architectures**: **AURORA-Scratch:** decoder-only Transformer pretraining (10M–30M parameters) from random initialisation. **AURORA-Adapt:** SFT and parameter-efficient fine-tuning (PEFT / QLoRA) on small open-weight base models.

**Evidence tags.** `[V]` read on the dataset page, card or paper · `[S]` read via a search-result copy (re-open and confirm before release) · `[A]` planning assumption, to be replaced by a measurement · `[U]` not yet verified (a to-do).

## Contents

1. Executive summary and core objectives
2. Master dataset comparison matrix
3. Elaborated dataset profiles (with a record per dataset)
4. Primary recommendation and strategic advantages
5. Governance, licensing and preprocessing protocol
6. Criteria-based comparison
7. CodeSearchNet versus the alternatives
8. Storage and preprocessing estimate
9. Pilot dataset recommendation

---

## 1. Executive Summary & Core Objectives

This report fulfils **Objective 1** of the AURORA project brief: establishing a reproducible, licence-compliant and compute-aware code dataset pipeline.

Due to hardware and storage constraints (zero-cost principles and student-level GPU quotas), we cannot naively download terabytes of raw code. We evaluated **eight dataset candidates** across multiple granularities (file-level, function-level, commit diffs and instruction pairs).

This document provides:

1. A complete comparative matrix across all candidate datasets (Section 2).
2. Individual dataset profiles detailing size, licensing, provenance and quality filters, each with a structured record: dataset name and version; source URL and access date; licence and usage conditions; processing and filtering steps; deduplication method; train/validation split method; manifest and content hashes (Section 3).
3. A recommendation of the top **3 core datasets** and their advantages over the remaining options (Section 4).
4. A governance protocol, criteria-based comparison, CodeSearchNet head-to-head, storage and preprocessing estimate and an explicit pilot dataset (Sections 5–9).

**Recommendation in brief.** Pilot on a ~300 MB Python slice of `codeparrot/github-code-clean` (streamed, licence-filtered, deduplicated, split by repository), with `codeparrot-clean` as fallback and `bigcode/the-stack-smol` for smoke tests. Use `CommitPackFT` plus one instruction set for SFT. Keep CodeSearchNet as a retrieval benchmark rather than the pretraining corpus.

**Three findings drive the design.**

- **No candidate is licensed uniformly.** Dataset-level labels (MIT, Apache-2.0) do not describe the code inside; per-file or per-repository licences vary and include GPL, AGPL, LGPL, MPL and EPL. A licence allow-list is part of the pipeline (Section 5.1).
- **Few sources state deduplication.** CodeSearchNet and CodeParrot Clean state it; `github-code-clean`, CommitPackFT and The Stack smol do not. Our own dedup is mandatory (Section 5.3).
- **Held-out data must come from the same pipeline.** `codeparrot-clean` and `github-code-clean` share their BigQuery origin, so one is not a safe validation set for the other. Validation and test data are split by repository from our own slice (Section 5.4).

---

## 2. Master Dataset Comparison Matrix

| Dataset / source name | Primary track target | Granularity | Storage / download size | Licence & provenance | Key preprocessing & filtering rules | Selected role in pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`bigcode/the-stack-smol`** (Python folder) | `AURORA-Scratch` | File-level | 10,000 files per language `[V]`; `ml6team` Python mirror 28.8 MB `[V]`; bigcode Python-only size `[U]` | No separate licence; original licences apply; **gated** terms `[V]`. Fields include repo and path | Random ~0.1% sample; line-length and alphanumeric values are columns, not applied filters `[V]` | **Pilot local smoke-testing** |
| **`codeparrot/github-code-clean`** (Python slice) | `AURORA-Scratch` | File / repo-level | **~1–5 GB** subset target (314 GB full corpus `[V]`) | Label apache-2.0 `[V]`; per-file licences mixed, tracked in `license` `[S]` | Drops auto-generated code, avg line length ≥ 100 and alphanumeric fraction ≤ 0.25 `[V]`; no dedup stated | **Primary causal pretraining corpus** |
| **`bigcode/commitpackft`** (Python) | `AURORA-Adapt` | Commit diffs & messages (single file) | Python 132.68 MB / 56,025 samples; all languages ~1.5 GB / 702,062 samples `[V]` | MIT label; per-sample `license` mixed (incl. agpl-3.0, lgpl-2.1, epl-1.0, mpl-2.0) `[V]`. Fields: `commit`, `repos`, `license` | Commit messages filtered to instruction-like patterns `[V]` | **SFT task & edit adaptation** |
| **`flytech/python-codes-25k`** | `AURORA-Adapt` | Instruction / code pairs | 24,813 entries `[S]` (≈12.6M tokens `[A]`) | MIT label `[S]`; provenance undocumented, appears AI-generated | Cleaning steps undocumented `[S]` | **SFT conversational task fine-tuning** (team decision) |
| **`bigcode/self-oss-instruct-sc2-exec-filter-50k`** (StarCoder2 Self-Alignment) | `AURORA-Adapt` | Instruction / execution pairs | ~50.7k rows `[S]` | ODC-By label `[S]`; seeds from The Stack v1 | Responses executed and filtered, then deduplicated `[S]` | **SFT execution-verified logic** |
| **`codeparrot/codeparrot-clean`** (train + valid) | Fallback / reference | File-level | Train 5,361,373 rows, ~12.8 GB `[V]`; valid 61,373 rows `[V]` | No dataset-level licence on card; per-file `license`, 15 values incl. gpl-3.0 `[V]` | Avg line < 100, max line < 1000, alnum > 0.25, auto-generated removed; exact-match dedup `[V]` | **Pilot fallback; filter-rule reference** (not a benchmark for `github-code-clean` models) |
| **`code-search-net/code_search_net`** (Python) | Retrieval / RAG | Function-level + docstrings | 581 MB download / 1.74 GB on disk `[V]` | Label "other"; per-repo licences, per-example licence not included `[V]`. Fields: repo, path, sha, url | Docs / length / name filters; duplicates and near-duplicates removed `[V]` | **RAG subsystem & vector-index benchmark** |
| **`ise-uiuc/Magicoder-OSS-Instruct-75K`** | Optional SFT | Synthetic code pairs | ~75K entries; 203 MB `[S]` | MIT label `[S]`; LLM-generated, generator unnamed in the paper `[V]` | Decontaminated; identical / same-seed duplicates removed `[V]` | **Secondary SFT augmentation (low priority)** |

---

## 3. Elaborated Dataset Profiles

Each profile keeps the focus / provenance / storage / pros-and-cons summary.

### 3.1 `bigcode/the-stack-smol` — Python folder

* **Focus & Scope:** 10,000 Python files per language, a ~0.1% random sample of The Stack, designed for small-scale language modelling `[V]`.
* **Provenance & Access:** Hugging Face (`bigcode/the-stack-smol`). **Gated**: the Terms of Use must be accepted. Includes repository and path metadata `[V]`.
* **Storage & Preprocessing:** the `ml6team/the-stack-smol-python` mirror is a single 28.8 MB parquet `[V]`; the Python-only size of the bigcode folder is `[U]`. Filters are *not* pre-applied (line-length and alphanumeric values are columns), so our own cleaning applies.
* **Pros & Cons:** Extremely fast download and low memory footprint. Too small to train a generalised generative model on its own.

**Dataset record**

| Item | Record |
|---|---|
| 1. Name and version | `bigcode/the-stack-smol`, subset `data/python`. Revision SHA not yet pinned. Python-only mirror `ml6team/the-stack-smol-python` also exists (28.8 MB single parquet, 10K–100K rows; licence, source version and fields not shown on its page) `[V]`; its provenance is unconfirmed, so it is not used for anything beyond a quick test. |
| 2. Source URL | https://huggingface.co/datasets/bigcode/the-stack-smol `[V]` |
| Access | **Needs permission** — gated; log in to Hugging Face and accept the Terms of Use before downloading `[V]`. |
| 3. Licence and conditions | No separate licence; use "must abide by the terms of the original licenses". **Gated:** accept the Terms of Use, update your copy to honour removal requests, pass the terms on if redistributing `[V]`. |
| 4. Processing / filtering | Random sample of 10,000 files per language (~0.1% of The Stack). No other filters described. `avg_line_length`, `max_line_length`, `alphanum_fraction` are columns, not applied filters `[V]`. |
| 5. Deduplication | Not stated; Stack version not stated `[V]`. Own dedup applies. |
| 6. Split | Single `train` split, 300,000 rows over all languages `[V]`. Own split applies. |
| 7. Manifest / hashes | Per-file SHA-256 manifest for Python files used (Section 5.5). Python-only size `[U]`. |

Use: pipeline smoke tests. Too small to train on alone.

### 3.2 `codeparrot/github-code-clean` — Python slice

* **Focus & Scope:** Large-scale open-source GitHub code (30 languages on the page summary), including Python `[V]`.
* **Provenance & Access:** Hugging Face (`codeparrot/github-code-clean`), no gating. Tracks a per-file `license` field `[S]`.
* **Storage & Preprocessing:** ~314 GB for the full repository `[V]`. The source removed 3.39M files (~2.94%) that were auto-generated, had long lines or a low alphanumeric fraction `[V]`.
* **Pros & Cons:** Strong pretraining candidate: file-level, quality-filtered, per-file licences. Requires streaming a targeted Python slice to fit local disk. No dedup is stated, the loader is a script, and size descriptions conflict (see below).

**Dataset record**

| Item | Record |
|---|---|
| 1. Name and version | `codeparrot/github-code-clean`, language `Python`. Revision SHA not yet pinned. Parent `codeparrot/github-code` states its BigQuery query was run on 14 Feb 2022 `[S]`. |
| 2. Source URL | https://huggingface.co/datasets/codeparrot/github-code-clean `[V]` |
| Access | **Easy download** — open, no permission needed `[V]`; the loader is a script, so if `datasets` refuses it, read the parquet shards directly. |
| 3. Licence and conditions | Label **apache-2.0** `[V]`; the loading script declares "Multiple: see the 'license' field" `[S]` and files include copyleft licences. The label does not cover the code. Apply the allow-list (5.1). No gating `[V]`. |
| 4. Processing / filtering | Cleaner version of `github-code`: average line length < 100; alphanumeric fraction > 0.25; auto-generated files removed by keyword search; 3.39M files (~2.94%) removed `[V]`. |
| 5. Deduplication | **Not stated** `[V]` (the raw README gives no counts or dedup). Own exact + near dedup required (5.3). |
| 6. Split | Single `train` split `[V]`. Own repo-level split (5.4). |
| 7. Manifest / hashes | Record revision SHA, filters used, per-file SHA-256 for the slice. |

Known issues: (a) size descriptions conflict — loader description "115M files, 32 languages" versus page summary ~11.0M rows, 30 languages, 314 GB `[V/S]`; the raw README states neither; confirm counts at download. (b) The repo uses a loading script that reads parquet shards `[S]`; recent `datasets` versions may refuse scripts, so test loading first and fall back to reading parquet directly. (c) Python share and slice size `[U]`.

### 3.3 `bigcode/commitpackft` — Python

* **Focus & Scope:** CommitPack (4 TB) filtered to ~2 GB: 702,062 samples in total, 56,025 in Python, commit messages that resemble natural instructions `[V]`.
* **Provenance & Access:** Hugging Face (`bigcode/commitpackft`), MIT label. Tracks `commit`, `old_file`, `new_file`, `repos`, `license` `[V]`.
* **Storage & Preprocessing:** Python 132.68 MB; ~1.5 GB for all languages `[V]`. Straightforward JSON field extraction (`old_contents` vs `new_contents`).
* **Pros & Cons:** Captures realistic software-engineering workflows (bug fixing, refactoring, features). Each sample is a single-file edit, and diffs can occasionally be fragmented.

**Dataset record**

| Item | Record |
|---|---|
| 1. Name and version | `bigcode/commitpackft`, `lang = Python`. Revision SHA not yet pinned. |
| 2. Source URL | https://huggingface.co/datasets/bigcode/commitpackft `[V]` |
| Access | **Easy download** — open, no permission needed `[V]`. |
| 3. Licence and conditions | Label MIT; repositories are described as permissively licensed, but per-sample `license` includes **agpl-3.0, lgpl-2.1, epl-1.0, mpl-2.0** and `unknown` besides mit, apache-2.0, bsd, isc, cc0-1.0, unlicense, artistic-2.0 `[V]`. Filter by this field (5.1). No gating `[V]`. |
| 4. Processing / filtering | Filtered version of CommitPack keeping instruction-like commit messages; criteria are in the OctoPack repository, not on the card `[V]`. Each sample is **one file** before and after a commit (`old_contents`, `new_contents`, `subject`, `message`, `license`, `repos`); not multi-file edits `[V]`. |
| 5. Deduplication | Not stated `[V]`. |
| 6. Split | No splits; the card's "Data Splits" is a per-language breakdown `[V]`. Own repo-level split (5.4). |
| 7. Manifest / hashes | Per-sample `commit` id plus SHA-256 of `new_contents`. |

Sizes: Python 56,025 samples / 132.68 MB; all languages 702,062 samples / 1545.02 MB (page footer shows 1.58 GB) `[V]`.

### 3.4 `flytech/python-codes-25k`

* **Focus & Scope:** 24,813 instructional Python tasks (`instruction`, `input`, `output`, `text`) `[S]`.
* **Provenance & Access:** Hugging Face (`flytech/python-codes-25k`), MIT label `[S]`. Content provenance is undocumented.
* **Storage & Preprocessing:** Lightweight; ~508 average tokens per example `[S]`, so ≈12.6M tokens in total `[A]`. The `text` field combines instruction, input and output.
* **Pros & Cons:** Highly structured for SFT. Only 25k tasks, so it needs to be combined with other data. Content appears AI-generated.

**Dataset record**

| Item | Record |
|---|---|
| 1. Name and version | `flytech/python-codes-25k`. Revision SHA not yet pinned. |
| 2. Source URL | https://huggingface.co/datasets/flytech/python-codes-25k (via card copy) `[S]` |
| Access | **Easy download** — open, no gating noted `[S]`. |
| 3. Licence and conditions | MIT label `[S]`. The card does not state where content comes from; the `input` field is described as the intro of an "AI response", so content appears model-generated. Provenance and provider terms `[U]`; to be approved before any release. |
| 4. Processing / filtering | Described as cleaned; fields `instruction`, `input`, `output`, `text`; 24,813 entries; ~508 average tokens `[S]`. Steps undocumented. |
| 5. Deduplication | Statistics only: 24,580 unique instructions, 24,581 unique outputs, 24,813 unique texts `[S]`. No method stated. |
| 6. Split | Single `train` split `[S]`. Own split by instruction hash (5.4). |
| 7. Manifest / hashes | SHA-256 of `instruction + output`; keep original row index. |

### 3.5 `bigcode/self-oss-instruct-sc2-exec-filter-50k` (StarCoder2 Self-Alignment)

* **Focus & Scope:** ~50.7k execution-filtered instruction–response pairs (hub viewer count) `[S]`.
* **Provenance & Access:** Hugging Face, ODC-By label `[S]`. Seeds are Python functions from The Stack v1; responses generated by StarCoder2 self-alignment `[S]`.
* **Storage & Preprocessing:** Standard prompt–response formatting. Download size not verified `[U]`.
* **Pros & Cons:** Every response was executed against its own checks, which removes broken code from the SFT mixture but does not prove correctness. Heavily focused on single-function algorithmic tasks.

**Dataset record**

| Item | Record |
|---|---|
| 1. Name and version | `bigcode/self-oss-instruct-sc2-exec-filter-50k`. Revision SHA not yet pinned. Exact row count: hub viewer shows ~50.7k; confirm at download. |
| 2. Source URL | https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k (via metadata and the StarCoder2-Instruct card) `[S]` |
| Access | **Easy download** — open, no gating noted `[S]`. |
| 3. Licence and conditions | odc-by `[S]` (attribution required); not restated on the model page. Seeds are Python functions from The Stack v1 (permissive) `[S]`. Responses are model-generated. |
| 4. Processing / filtering | Seed functions from The Stack v1 → judged by StarCoder2-15B to drop weak docstrings → concepts → instructions → responses (~500k raw) → **execution filter** → dedup → ~50k `[S]`. |
| 5. Deduplication | Final-step dedup stated; method not on the pages read `[S]`. |
| 6. Split | `train` only `[S]`. Own split (5.4). |
| 7. Manifest / hashes | SHA-256 of `instruction + response`. |

Execution filtering shows responses ran against their own checks; it does not prove functional correctness. Passing tests is evidence, not proof.

### 3.6 `codeparrot/codeparrot-clean` (train and valid)

* **Focus & Scope:** Cleaned Python files; train has 5,361,373 rows and the separate validation repo `codeparrot-clean-valid` has 61,373 rows `[V]`.
* **Provenance & Access:** Hugging Face (`codeparrot/codeparrot-clean`; card links `lvwerra/codeparrot-clean`). Includes `repo_name`, `path`, `copies`, `license`, `hash` `[V]`.
* **Storage & Preprocessing:** ~12.8 GB listed for train `[V]`; validation size `[U]`. Exact-match dedup only.
* **Pros & Cons:** A convenient, ready-made validation split and a reference for filter rules. Because it shares its BigQuery origin with `github-code-clean`, it is not a guaranteed leakage-free benchmark for models trained on that corpus.

**Dataset record**

| Item | Record |
|---|---|
| 1. Name and version | `codeparrot/codeparrot-clean`; the card links the same data as `lvwerra/codeparrot-clean`. Canonical id to confirm at download. Revision SHA not yet pinned. |
| 2. Source URL | https://huggingface.co/datasets/codeparrot/codeparrot-clean and https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid `[V]` |
| Access | **Easy download** — open, no permission needed `[V]`. |
| 3. Licence and conditions | **No dataset-level licence on the card.** Per-file `license` column with 15 values including gpl-3.0 `[V]`. Apply 5.1. No gating `[V]`. |
| 4. Processing / filtering | Average line length < 100; maximum line length < 1000; alphanumeric fraction > 0.25; auto-generated files removed by keyword search `[V]`. Fields: `repo_name, path, copies, size, content, license, hash, line_mean, line_max, alpha_frac, autogenerated` `[V]`. |
| 5. Deduplication | Exact-match removal only `[V]`; near-duplicates may remain. |
| 6. Split | Train (5,361,373 rows) and a separate validation repo `codeparrot-clean-valid` (61,373 rows) `[V]`. The method used to create the split is not described. |
| 7. Manifest / hashes | A per-file `hash` field exists `[V]`; we still compute our own SHA-256. |

Used as the pilot fallback and for the leakage check (5.4). Not used as the held-out set for a model trained on `github-code-clean`.

### 3.7 `code-search-net/code_search_net` — Python

* **Focus & Scope:** ~457k Python functions paired with docstrings (457,461 across train / valid / test) `[V]`.
* **Provenance & Access:** Hugging Face and GitHub (`github/CodeSearchNet`). The tool repo is MIT; the data carries per-repository licences `[V]`. Preserves repo, path, sha and url fields.
* **Storage & Preprocessing:** Python 581 MB download / 1.74 GB on disk `[V]`. Function-level parsing already done.
* **Pros & Cons:** Standard benchmark for semantic code retrieval and RAG. Code dates from 2019 (lacks newer Python 3.10+ syntax). No per-example licence.

**Dataset record**

| Item | Record |
|---|---|
| 1. Name and version | `code-search-net/code_search_net`, config `python`. Revision SHA not yet pinned. Also on Zenodo (record 7908468) `[V]`. |
| 2. Source URL | https://huggingface.co/datasets/code_search_net · https://github.com/github/CodeSearchNet · paper https://arxiv.org/abs/1909.09436 `[V]` |
| Access | **Easy download** — open, no permission needed `[V]`. |
| 3. Licence and conditions | Label "other": each repository has its own licence; per-example licence **not included**, so users determine licences themselves `[V]`. The MIT licence belongs to the tool repository, not the data. Repositories without a licence that permits redistribution were removed `[V]`. |
| 4. Processing / filtering | Public non-fork GitHub repos selected via libraries.io (used by at least one other project, sorted by stars and forks) `[V]`; Tree-sitter parsing; removal of undocumented functions, docstrings < 3 tokens, functions < 3 lines, "test" in name, constructors and extension methods; documentation cut to the first paragraph `[V]`. |
| 5. Deduplication | "Remove duplicates … by identifying (near) duplicate functions and only keeping one copy" following Allamanis and Lopes et al.; no threshold given (paper Section 2) `[V]`. |
| 6. Split | Provided splits 412,178 / 23,107 / 22,176 (Python) `[V]`. The paper states 80-10-10 and does not say whether splits are by repository `[V]`. |
| 7. Manifest / hashes | HF parquet files (train 521.8 MB, valid 30.7 MB, test 28.7 MB) `[V]`; record file checksums and SHA-256 of `func_code_string`. |

Sizes: 581 MB download, 1.74 GB on disk for Python `[V]`.

### 3.8 `ise-uiuc/Magicoder-OSS-Instruct-75K` (optional)

* **Focus & Scope:** ~75K LLM-generated instruction–code pairs seeded from open-source snippets `[V]`.
* **Provenance & Access:** Hugging Face, MIT label `[S]`. Generator is described only as "an LLM (e.g., ChatGPT)" `[V]`.
* **Storage & Preprocessing:** 203 MB jsonl `[S]`.
* **Pros & Cons:** Decontaminated against common benchmarks. Model-generated, so hallucination and provider-terms risks apply; low priority.

**Dataset record**

| Item | Record |
|---|---|
| 1. Name and version | `ise-uiuc/Magicoder-OSS-Instruct-75K`. Revision SHA not yet pinned. |
| 2. Source URL | https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K `[S]`; paper https://arxiv.org/abs/2312.02120 (read via mirror) `[V]` |
| Access | **Easy download** — open, no gating noted `[S]`. |
| 3. Licence and conditions | MIT label `[S]`. Content is LLM-generated; the paper names the generator only as "an LLM (e.g., ChatGPT)", so the exact model and its terms are not confirmed `[V]`. |
| 4. Processing / filtering | ~80K generated from seed snippets (starcoderdata) reduced to ~75K entries; decontaminated against HumanEval, MBPP, APPS, DS-1000 and GSM8K (only 9 further samples removed) `[V]`. |
| 5. Deduplication | Identical samples or samples sharing a seed snippet removed `[V]`. |
| 6. Split | Single file, no splits `[S]`. |
| 7. Manifest / hashes | 203 MB `data-oss_instruct-decontaminated.jsonl`; record its SHA-256. |

---

## 4. Primary Recommendation & Strategic Advantages

### Selected top-3 core dataset strategy

To maximise efficiency and meet the AURORA milestone without exceeding GPU or storage constraints, we select **3 core datasets**:

1. **Pretraining core:** Python subset of `github-code-clean` (pilot ~300 MB; full target per Section 8).
2. **Instruction / SFT core:** `CommitPackFT` (Python split) plus one instruction set (`Python-Codes-25K` or `self-oss-instruct`, see the table below).
3. **Retrieval core:** `CodeSearchNet` (Python subset), for evaluation and RAG rather than training.

### Key advantages of the selected stack over alternatives

#### 1. Pretraining quality: `github-code-clean` vs raw repositories / `CodeSearchNet`

* **Advantage:** Raw GitHub code contains auto-generated files, minified scripts and repetitive boilerplate that waste parameter capacity in small (10M–30M) models. `github-code-clean` has already removed auto-generated files and long-line files `[V]`.
* **Versus CodeSearchNet:** CodeSearchNet holds isolated, documented functions. Pretraining `AURORA-Scratch` on whole files lets the model learn imports, module structure and multi-function context (detail in Section 7).

#### 2. Instruction tuning: `CommitPackFT` + one instruction set vs relying on synthetic data alone

* **Advantage:** `CommitPackFT` captures human developer intent through real commit messages and diffs. Synthetic instruction sets (`OSS-Instruct`, `Python-Codes-25K`, `self-oss-instruct`) carry hallucination and provider-terms risks, so each is chosen on documented provenance, licence clarity and verification, not on whether it is synthetic.
* **Coverage:** `CommitPackFT` supplies single-file edit tasks; an instruction set supplies prompt-to-code generation.

#### 3. Storage and compute efficiency vs full open-code corpora (e.g. The Stack v2 / RedPajama)

* **Advantage:** Full open-code corpora require terabytes of storage and weeks of bandwidth. The selected stack allows:
  * Local pipeline verification via `the-stack-smol` (Python).
  * Streaming pretraining via a targeted slice of `github-code-clean`.
  * Fast instruction fine-tuning via QLoRA on a small SFT mixture.

### Selected stack and conditions

| Role | Dataset | Condition |
|---|---|---|
| Pretraining (Scratch) | `github-code-clean`, Python slice (pilot ~300 MB; full target per Section 8) | Own dedup and licence filter mandatory |
| SFT (Adapt) | `commitpackft` (Python) | Filter by per-sample licence |
| SFT (Adapt), team decision | `python-codes-25k` **or** `self-oss-instruct-sc2-exec-filter-50k` | Both model-generated. The second has a documented pipeline, ODC-By licence and execution filtering; the first has undocumented provenance. Recommendation: prefer the second unless the team accepts the first's provenance risk. |
| Retrieval / evaluation | CodeSearchNet (Python) | Licences cannot be filtered per example; use for evaluation, not training |
| Smoke tests | `the-stack-smol` (Python) | Accept gated terms |
| Reference / fallback | `codeparrot-clean` | Not the held-out set for `github-code-clean` models |
| Optional | Magicoder OSS-Instruct 75K | Generator and terms unconfirmed |

Model-generated sets share one risk: they are chosen on documented provenance, licence clarity and verification, not on whether they are synthetic.

---

## 5. Governance, Licensing, and Preprocessing Protocol

### 5.1 Provenance and licence checks

For every dataset ingested into the project:

1. **Licence verification:** dataset-level labels for the core datasets are permissive (**MIT**, **Apache-2.0**, **ODC-By**), but the code inside is mixed, so each file's licence is checked against the allow-list below.
2. **Metadata retention:** original provenance fields (`repo`, `path`, `commit`, `sha`) are preserved in the preprocessed JSONL manifests (`data/manifest.py`).

- Record the dataset-level label and each file's licence in the manifest.
- Allow-list for training data: `mit`, `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `isc`, `unlicense`, `cc0-1.0`.
- Excluded by default: `gpl-*`, `agpl-*`, `lgpl-*`, `mpl-*`, `epl-*`, `unknown` / missing, and anything not on the list.
- Permissive licences still require attribution: keep `repo_name`, `path` and `license` for every training file.
- Datasets with extra conditions: `the-stack-smol` (gated; removal-request terms), `self-oss-instruct` (ODC-By attribution), model-generated sets (provider terms `[U]`).
- This is a technical proposal, not legal advice.

### 5.2 Processing steps

1. Pin the dataset revision and record the access date.
2. Select language and licence subset.
3. Normalise: valid UTF-8, consistent line endings; trailing-whitespace stripping applies to hashing only.
4. Apply the same quality filters to every dataset (average line length < 100, maximum line length < 1000, alphanumeric fraction > 0.25, drop auto-generated files).
5. Deduplicate (5.3), split (5.4), write the manifest (5.5).

### 5.3 Deduplication

1. **Exact:** SHA-256 of the normalised text; keep the first occurrence in `(repo_name, path)` order.
2. **Near-duplicate:** MinHash + LSH over token shingles; Jaccard threshold starts at ~0.85 and is tuned on a sample `[A]`. Report files removed per stage.
3. **Cross-set:** drop any training file whose hash or near-duplicate appears in validation, test or evaluation benchmarks (HumanEval, MBPP).
4. Report duplicate rates per dataset, as the proposal's data-pipeline section requires.

### 5.4 Train / validation / test split

- Split by repository: assign each `repo_name` to a split by hash with a fixed seed (proposed 98 / 1 / 1), so files from one repository never cross splits.
- Instruction datasets without repository fields: split by hash of the instruction text.
- Keep CodeSearchNet's provided splits for retrieval benchmarking.
- Do not use `codeparrot-clean` validation as held-out data for a model trained on `github-code-clean` unless a hash overlap check shows no overlap.
- Keep HumanEval and MBPP out of training.

### 5.5 Manifest and hashes

Dataset-level `manifest.json`: dataset id, URL, revision SHA, access date, licence allow-list, filter parameters, row counts after each stage, SHA-256 of the manifest. Per-file `manifest.jsonl`, one line per kept file:

```json
{"dataset": "codeparrot/github-code-clean", "revision": "<sha>", "repo_name": "...", "path": "...",
 "license": "mit", "sha256": "<hash of normalised content>", "size_bytes": 0, "split": "train"}
```

Pin a revision:

```python
from huggingface_hub import HfApi
info = HfApi().dataset_info("codeparrot/github-code-clean")
print(info.sha, info.last_modified)
```

For full corpora only the revision SHA and shard checksums are recorded.

### 5.6 Storage discipline

The full `github-code-clean` repository (~314 GB) is not stored locally. The pipeline streams a targeted Python subset (pilot ~300 MB; full target in Section 8) so the work stays reproducible within student compute limits.

---

## 6. Criteria-Based Comparison

Scores: 3 = strong, 2 = acceptable with work, 1 = weak, 0 = disqualifying for this use. Scores are the author's judgement from the evidence above and are meant to be challenged by the reviewers.

| Criterion | github-code-clean | codeparrot-clean | the-stack-smol | CodeSearchNet | CommitPackFT | Instruction sets |
|---|---|---|---|---|---|---|
| Per-file licence available | 3 | 3 | 3 | 0 (not included) | 3 | 1–2 |
| Source states dedup | 0 | 2 (exact only) | 0 | 3 | 0 | 1–3 (varies) |
| Fits laptop / free-tier disk | 2 (streamed slice) | 2 | 3 | 3 | 3 | 3 |
| Python-heavy, easy to filter | 3 | 3 | 3 | 3 | 3 | 3 |
| Fit for from-scratch pretraining | 3 | 3 | 1 (too small) | 1 (functions only) | 1 | 0 |
| Fit for SFT | 0 | 0 | 0 | 1 | 3 | 3 |
| Provenance documented | 2 | 2 | 2 | 3 | 3 | 1–2 |
| Access friction | 1–2 (loader script) | 3 | 1 (gated) | 3 | 3 | 3 |

Reading: `github-code-clean` wins on size and metadata but needs the most of our own work (dedup, licence filter). `codeparrot-clean` is the lowest-friction fallback. `the-stack-smol` is the right size for tests only.

---

## 7. CodeSearchNet versus the Alternatives

CodeSearchNet is a well-documented dataset, but it was built for a different job: matching natural-language queries to functions.

- **Granularity.** Records are single functions with a docstring. Functions without documentation, with docstrings under three tokens, under three lines, with "test" in the name, constructors and standard extension methods are all removed `[V]`. A code model trained only on this never sees whole files, imports, classes in context or test code.
- **Age.** The corpus dates from 2019 (paper arXiv 1909.09436) `[V]`, so newer Python syntax and libraries are underrepresented.
- **Licence.** The card label is "other"; each repository has its own licence and per-example licence information is not included `[V]`. Repositories without a redistribution-permitting licence were removed at collection `[V]`, which is good, but our allow-list cannot be applied per example.
- **What it does well.** The card and the paper state that duplicates and near-duplicates were removed `[V]`. The paper cites Allamanis and Lopes et al. for the method and gives no similarity threshold `[V]`. Train / validation / test splits are provided (Python 412,178 / 23,107 / 22,176, i.e. about 90 / 5 / 5) `[V]`. The paper states an 80-10-10 split overall and does not say whether splits are by repository `[V]`; the Python proportions above therefore differ from the paper's stated ratio and should be treated as the authoritative numbers for the HF version.
- **Conclusion.** Use CodeSearchNet for retrieval / RAG evaluation and docstring-to-code tasks. Use a file-level corpus for pretraining.

---

## 8. Storage and Preprocessing Estimate

All figures marked `[A]` are planning assumptions to be replaced by measurements on a 10k-file sample.

**Assumptions.**

- Token budget `[A]`: ~20 training tokens per parameter, so 200M tokens (10M params) to 600M tokens (30M params).
- Compression `[A]`: 3–4 characters per token for code with byte-level BPE; replace with a measured value. Cleaned text is roughly 1 byte per character, so cleaned size ≈ tokens × 3–4 bytes: **~0.6–2.4 GB**.
- Dedup loss `[A]`: when the source states no dedup, budget the raw slice at ~2× the target. (Data point: near-dedup of The Stack v1 roughly halved it, 3 TB → 1.5 TB, per a W&B write-up.)
- Working space `[A]`: ~3× the slice (raw, filtered, deduplicated copies).

| Step | github-code-clean (pilot) | github-code-clean (full target) | CommitPackFT (Python) | CodeSearchNet (Python) | the-stack-smol (Python) |
|---|---|---|---|---|---|
| Source size | 314 GB total, never downloaded `[V]` | same | 132.68 MB `[V]` | 581 MB download `[V]` | 28.8 MB mirror `[V]`; bigcode Python folder `[U]` |
| Raw slice to stream `[A]` | ~600 MB | ~1.2–4.8 GB | 132.68 MB | 581 MB | ~10k files |
| After licence + quality filters | measure on 10k files | measure | measure | n/a (no licence field) | measure |
| After dedup `[A]` | ~300 MB | 0.6–2.4 GB | measure | already deduplicated `[V]` | measure |
| Tokens at that size `[A]` | ~75–100M | 200–600M | ~35–45M (132.68 MB at 3–4 bytes/token) | n/a | n/a |
| Local disk including working copies `[A]` | ~1–2 GB | ~2–7 GB | < 1 GB | ~2 GB | < 1 GB |
| Where to run | laptop or free notebook | free notebook or club GPU (quota to verify, proposal §11.1) | laptop | laptop | laptop |

**Preprocessing time.** Not estimated from memory. Measure on a 10k-file sample (parse, filter, hash, MinHash) and extrapolate; record the numbers in the pipeline report.

**Preprocessing steps and expected cost drivers.** UTF-8 validation and filters are linear and cheap. Exact dedup is a hash pass. MinHash + LSH is the expensive step and dominates memory; run it per language on the slice, not on the full corpus. Hashing every file for the manifest is practical at 1–5 GB.

---

## 9. Pilot Dataset Recommendation

**Pilot: ~300 MB Python slice of `codeparrot/github-code-clean`.**

Why:

- File-level code with `repo_name`, `path` and per-file `license` fields, so allow-listing and attribution are possible `[S]`.
- Quality filters already applied by the source (line length, alphanumeric fraction, auto-generated) `[V]`.
- Streamable, so the 314 GB repository is never stored locally.
- ~75–100M tokens `[A]` is small enough to rerun the whole data pipeline many times.

**Reproducible sampling rule.** Stream Python rows, keep those whose `license` is in the allow-list (5.1), order by SHA-256 of `repo_name + "/" + path`, and keep the first N files until the cleaned slice reaches ~300 MB. The same revision SHA and the same rule must give the same manifest.

**Split.** Repository-level hash split, 98 / 1 / 1, fixed seed (5.4).

**Fallback.** A Python slice of `codeparrot/codeparrot-clean` (exact-deduplicated, per-file licence, no loader script, no gating) if `github-code-clean` cannot be loaded. **Smoke tests:** `bigcode/the-stack-smol` Python (10,000 files), after accepting its terms.

**Acceptance tests (pilot is accepted when all pass).**

1. The loader streams 1,000 Python files, each with a `license` value.
2. File counts after licence filter, quality filter, exact dedup and near dedup are reported.
3. Rerunning the pipeline at the same revision produces identical manifest hashes.
4. Overlap check against HumanEval and MBPP reports zero exact matches.

Instruction-tuning data is not part of the pilot; it is added in the Adapt track (Section 4).
