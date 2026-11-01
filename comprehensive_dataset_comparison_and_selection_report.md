# AURORA — Comprehensive Dataset Research, Comparison, and Selection Report

**Document Status:** Finalized / Ready for Repository Integration  
**Primary Domain:** Code Data Pipeline, Dataset Curation, and Governance  
**Target Architectures:** 
* **AURORA-Scratch:** Decoder-only Transformer pretraining (10M–30M parameters) from random initialization.
* **AURORA-Adapt:** Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT / QLoRA) on small open-weight base models.

---

## 1. Executive Summary & Core Objectives

This report fulfills **Objective 1** of the AURORA project brief: establishing a reproducible, license-compliant, and compute-aware code dataset pipeline. 

Due to hardware and storage constraints (working under zero-cost principles and student-level GPU quotas), we cannot naively download terabytes of raw code data. We conducted a systematic evaluation of **eight major dataset candidates** across multiple granularities (file-level, function-level, commit diffs, and instruction pairs).

This document provides:
1. A complete comparative matrix across all candidate datasets.
2. Individual dataset profiles detailing size, licensing, provenance, and quality filters.
3. A strategic recommendation of the top **3 core datasets** and their concrete advantages over the remaining options.

---

## 2. Master Dataset Comparison Matrix

| Dataset / Source Name | Primary Track Target | Granularity | Storage / Download Size | License & Provenance | Key Preprocessing & Filtering Rules | Selected Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`the-stack-smol-python`** | `AURORA-Scratch` | File-level | **82.1 MB** (~28.7 MB download) | Permissive (The Stack) / Metadata: `repository_name`, `path` | Pre-filtered for avg/max line length and alphanumeric fraction ($\ge 0.25$). | **Pilot Local Smoke-Testing** |
| **`github-code-clean`** (Python Slice) | `AURORA-Scratch` | File / Repo-level | **~1 GB–5 GB** subset target (~314 GB full corpus) | Apache 2.0 / Per-file repository licenses tracked | Drops auto-generated code, files with avg line length $\ge 100$, and low alpha fraction. | **Primary Causal Pretraining Corpus** |
| **`CommitPackFT`** (Python Split) | `AURORA-Adapt` | Commit Diffs & Messages | **~1.54 GB** total (~56k Python commit samples) | MIT License / Metadata: `commit`, `repos`, `license` | Filtered commit messages matching natural language instruction patterns across 277 languages. | **SFT Task & Edit Adaptation** |
| **`Python-Codes-25K`** | `AURORA-Adapt` | Instruction / Code Pairs | **< 500 MB** (24,813 entries) | MIT License / Curated instruction-output fields | Clean Python code snippets paired with user prompts and task inputs. | **SFT Conversational Task Fine-Tuning** |
| **`StarCoder2 Self-Alignment`** | `AURORA-Adapt` | Concept / Execution Pairs | **~261 MB** (~90 MB download) | ODC-By / MultiPL-T seed functions | Execution-validated responses ensuring 100% functional correctness out-of-the-box. | **SFT Execution-Verified Logic** |
| **`CodeParrot Clean (valid)`** | Evaluation | File-level | **~50 MB–200 MB** (61,373 rows) | Apache 2.0 / Metadata: `repo_name`, `path`, `copies` | Deduplicated split tracking duplicate copy counts and complexity metrics. | **Held-Out Model Perplexity Benchmark** |
| **`CodeSearchNet`** (Python) | Retrieval / RAG | Function-level + Docstrings | **~3.5 GB** compressed (~20 GB all languages) | MIT License / Metadata: `repo`, `path`, `sha`, `url` | 2M parallel `(comment, code)` pairs across 6 languages with human relevance judgements. | **RAG Subsystem & Vector Index Benchmark** |
| **`OSS-Instruct`** | Optional SFT | Synthetic Code Pairs | **< 1 GB** (Synthetic) | MIT (OpenAI Usage Policy restricted) | Generated via `gpt-3.5-turbo-1106` targeting open-source problem descriptions. | **Secondary SFT Augmentation (Low Priority)** |

---

## 3. Elaborated Dataset Profiles & Audits

### 3.1 `the-stack-smol-python`
* **Focus & Scope:** 10,000 curated Python files designed for small-scale language modeling.
* **Provenance & Access:** Open access via Hugging Face (`bigcode/the-stack-smol-python`). Includes `repository_name` and `path`.
* **Storage & Preprocessing:** ~28.7 MB download / ~82.1 MB dataset size. Zero additional cleaning required.
* **Pros & Cons:** Extremely fast download and low memory footprint. However, it is too small to train a generalized generative model on its own.

### 3.2 `github-code-clean` (Python Slice)
* **Focus & Scope:** Large-scale open-source GitHub code across 32 languages, heavily weighted toward Python.
* **Provenance & Access:** Open access via Hugging Face (`codeparrot/github-code-clean`). Tracks per-file repository licenses.
* **Storage & Preprocessing:** Full dataset is ~314 GB compressed (115M files). Pipeline filters out 3.39M auto-generated or low-quality files (avg line length $\ge 100$, alphanumeric fraction $\le 0.25$).
* **Pros & Cons:** The highest quality pretraining corpus available for code LLMs. Requires streaming or downloading a targeted 1 GB–5 GB Python slice to fit local disk limits.

### 3.3 `CommitPackFT` (Python Split)
* **Focus & Scope:** 4TB CommitPack dataset filtered down to 2GB (~702k total samples, ~56k Python samples) containing commit messages that resemble natural instructions.
* **Provenance & Access:** Open access via Hugging Face (`bigcode/commitpackft`) under MIT License. Tracks `commit`, `old_file`, `new_file`, and `repos`.
* **Storage & Preprocessing:** ~1.54 GB total. Straightforward JSON field extraction required (`old_contents` vs `new_contents`).
* **Pros & Cons:** Captures realistic software engineering workflows (bug fixing, refactoring, and feature additions). Commit diffs can occasionally be fragmented.

### 3.4 `Python-Codes-25K`
* **Focus & Scope:** 24,813 clean instructional Python tasks covering code generation, text-to-code, and code-to-text.
* **Provenance & Access:** Open access via Hugging Face (`flytech/python-codes-25k`) under MIT License.
* **Storage & Preprocessing:** Lightweight (< 500 MB). Features unified `text` field combining `instruction`, `input`, and `output`.
* **Pros & Cons:** Highly structured for Supervised Fine-Tuning. Scope is restricted to 25k tasks, requiring combination with other datasets.

### 3.5 `StarCoder2 Self-Alignment`
* **Focus & Scope:** 50,661 execution-validated instruction-code responses derived from MultiPL-T seed functions.
* **Provenance & Access:** Open access via Hugging Face (`bigcode/starcoder2-self-alignment`) under ODC-By License.
* **Storage & Preprocessing:** ~90 MB download / ~261 MB uncompressed. Standard prompt-response formatting.
* **Pros & Cons:** Every sample is verified via execution, guaranteeing zero broken code in the SFT mixture. Heavily focused on single-function algorithmic tasks.

### 3.6 `CodeParrot Clean (valid)`
* **Focus & Scope:** 61,373 cleaned Python files used as a standardized validation split.
* **Provenance & Access:** Open access via Hugging Face (`lvwerra/codeparrot-clean`). Includes `repo_name`, `path`, and `copies`.
* **Storage & Preprocessing:** ~50 MB–200 MB. Fully deduplicated.
* **Pros & Cons:** Ideal held-out benchmark to measure true validation loss and perplexity without data leakage.

### 3.7 `CodeSearchNet` (Python)
* **Focus & Scope:** ~450k Python functions paired with top-level docstrings (2M samples across 6 languages).
* **Provenance & Access:** Hosted on GitHub and S3 public mirrors (`github/CodeSearchNet`) under MIT License. Preserves `repo`, `path`, `sha`, and `url`.
* **Storage & Preprocessing:** ~3.5 GB compressed (~20 GB total). Function-level parsing required.
* **Pros & Cons:** Standard benchmark for semantic code retrieval and RAG. Code dates back to 2019 (lacks newer Python 3.10+ syntax).

---

## 4. Primary Recommendation & Strategic Advantages

### Selected Top 3 Core Dataset Strategy

To maximize efficiency and fulfill the AURORA milestone without exceeding GPU or storage constraints, we select the following **3 Core Datasets**:

1. **Pretraining Core:** Filtered Python Subset of `github-code-clean` (1 GB–5 GB slice)
2. **Instruction/SFT Core:** `CommitPackFT` (Python split) + `Python-Codes-25K` (Hybrid Blend)
3. **Retrieval Core:** `CodeSearchNet` (Python subset)

---

### Key Advantages of this Selected Stack Over Alternatives

#### 1. Pretraining Quality: `github-code-clean` vs. Raw Repositories / `CodeSearchNet`
* **Advantage:** Raw GitHub code contains auto-generated lock files, minified scripts, and repetitive boilerplate that wastes parameter capacity in small models (10M–30M). `github-code-clean` has already removed auto-generated code and long lines ($\ge 100$ chars).
* **Superiority over CodeSearchNet:** CodeSearchNet only contains isolated functions. Pretraining `AURORA-Scratch` on full files from `github-code-clean` allows the model to learn repository structure, multi-function dependencies, and module imports.

#### 2. Instruction Tuning: `CommitPackFT` + `Python-Codes-25K` vs. Synthetic Data (`OSS-Instruct`)
* **Advantage:** Synthetic instruction datasets like `OSS-Instruct` (generated via GPT-3.5) carry hallucination risks and licensing constraints under OpenAI's usage policy.
* **Superiority:** `CommitPackFT` captures human developer intent via real commit messages and diffs, while `Python-Codes-25K` provides clean prompt-to-code structures under permissive MIT licenses. Combining them covers both multi-file edits and single-prompt code generation.

#### 3. Storage & Compute Efficiency vs. Full Open-Code Corpora (e.g., Full Stack v2 / RedPajama)
* **Advantage:** Full open-code corpora require terabytes of storage and weeks of bandwidth. Our selected 3-core stack allows:
  * Local pipeline verification via `the-stack-smol-python` (~82 MB).
  * Streaming pretraining via a 1 GB–5 GB slice of `github-code-clean`.
  * Fast instruction fine-tuning (< 2 GB total) via QLoRA.

---

## 5. Governance, Licensing, and Preprocessing Protocol

### 5.1 Provenance and License Auditing
For every dataset ingested into the project:
1. **License Verification:** All core datasets use open-source permissive licenses (**MIT**, **Apache 2.0**, or **ODC-By**).
2. **Metadata Retention:** Original file provenance fields (`repo`, `path`, `commit`, `sha`) must be preserved in the preprocessed JSONL manifests (`data/manifest.py`).

### 5.2 Storage Discipline
The full `github-code-clean` repository (~314 GB) will **not** be stored locally in its entirety. The dataset pipeline will stream a targeted **1 GB–5 GB Python subset** to ensure reproducibility within student compute limits.