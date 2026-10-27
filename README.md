# AURORA — A Ternary-Weight Coding Language Model

> **Project status:** Early development and research prototyping  
> **Primary language:** Python  
> **Deep learning framework:** PyTorch  
> **Research direction:** Coding-focused autoregressive language model with BitNet b1.58-inspired ternary weights  
> **Team size:** 7 members  
> **Current priority:** Establish the architecture, dataset pipeline, quantization specification, and reproducible training foundation

---

## 1. Project Overview

AURORA is a student-led research and engineering project to build a compact, compute-efficient language model specialized in code generation and code completion.

The project aims to understand and implement the complete language-model development pipeline rather than relying exclusively on externally hosted language models. This includes:

- Transformer architecture implementation.
- Ternary-weight quantization and low-bit training research.
- Dataset curation, tokenization, and preprocessing.
- Autoregressive language-model pretraining.
- Training infrastructure, checkpointing, and reproducibility.
- Code-generation evaluation.
- Integration with an existing coding-agent prototype.
- Future exploration of adaptive computation and efficient inference.

The project uses PyTorch for tensor operations, automatic differentiation, and training infrastructure. The core model architecture and quantization components will be implemented by the team to develop a deep understanding of the underlying methods.

### Primary objective

Build and evaluate a small coding-focused language model whose principal learned parameters use a ternary-weight representation inspired by BitNet b1.58.

### Secondary objectives

1. Establish a reproducible baseline using a conventional floating-point Transformer.
2. Implement and validate ternary-weight training.
3. Compare model quality, memory consumption, throughput, and training stability.
4. Investigate whether the model can support practical code generation and completion.
5. Integrate the model with an existing coding-agent prototype after the standalone model is functional.
6. Produce documented experiments that can justify future access to larger GPU resources.

### What AURORA is not

- It is not initially an attempt to train a frontier-scale language model.
- It is not merely a wrapper around a third-party LLM API.
- It is not assumed that ternary weights automatically provide faster inference.
- It is not considered successful solely because training loss decreases.
- It does not claim novelty for established techniques such as Transformers, quantization, retrieval-augmented generation, or model distillation without evidence of a distinct contribution.

---

## 2. Research Direction: BitNet b1.58

A central design goal is to investigate a ternary-weight architecture inspired by Microsoft's BitNet b1.58.

The term *1.58-bit* refers to the information-theoretic lower bound for representing three equally likely values:

\[
\log_2(3) \approx 1.585
\]

The effective quantized weights are restricted to:

\[
W_q \in \{-1,0,+1\}
\]

This does not mean that an ordinary PyTorch tensor containing these values automatically uses 1.58 bits per weight. Physical storage, arithmetic kernels, activation precision, and training state are separate implementation concerns.

### Research reference

Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*.

Paper: https://arxiv.org/abs/2402.17764

Reference implementation: https://github.com/microsoft/BitNet

### Initial approach

AURORA will develop two experimental paths:

**A. Floating-point reference model**

A small decoder-only Transformer used to verify the architecture, training loop, data pipeline, loss calculation, and checkpointing.

**B. Ternary-weight model**

A model using ternary-weight layers according to a documented quantization and training specification derived from the research paper.

The floating-point model is a correctness reference, not a decision to abandon the ternary-weight objective.

### Quantization principles

- The forward computation must use the intended quantized weights in the layers designated for ternary computation.
- Training may require higher-precision master weights so that an optimizer can update trainable parameters.
- Rounding is not ordinarily differentiable; the gradient approximation or training method must be explicitly designed and tested.
- Scaling and normalization must follow a documented method.
- Embeddings, normalization parameters, output layers, and activations must have their precision policy explicitly specified.
- Quantization correctness, training effectiveness, and hardware efficiency must be measured separately.

A simple ternarization equation may be used for an initial educational prototype, but it must not be represented as a faithful reproduction of BitNet b1.58 unless its method matches the paper.

### Important unresolved decisions

The following must be documented before claiming a faithful implementation:

- Exact weight quantization and scaling rule.
- Gradient strategy for ternary weights.
- Which layers use ternary weights.
- Activation quantization and precision policy.
- Initialization and normalization choices.
- Training optimizer and stability requirements.
- Whether optimized low-bit inference kernels are in scope for the first prototype.

---

## 3. Model Architecture

AURORA initially uses a decoder-only autoregressive Transformer.

Given a sequence of tokens, the model predicts the next token at each position.

The conceptual pipeline is:

1. Source code is filtered and tokenized.
2. Token IDs are grouped into fixed-length sequences.
3. Token and positional information are represented in the model.
4. Causal self-attention allows each position to attend only to permitted previous positions.
5. Feed-forward layers transform the hidden representations.
6. Residual connections and normalization support stable information flow.
7. The language-model head produces next-token logits.
8. Cross-entropy loss compares predictions with the target tokens.
9. Backpropagation and the optimizer update the trainable parameters.

The core objective is next-token prediction:

\[
\mathcal{L}
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log P_\theta(x_t \mid x_{<t})
\]

Here, \(x_t\) is the target token, \(x_{<t}\) represents the preceding context, and \(\theta\) denotes the trainable model parameters.

### Important tensor conventions

- Input token IDs: `[batch_size, sequence_length]`
- Target token IDs: `[batch_size, sequence_length]`
- Model logits: `[batch_size, sequence_length, vocab_size]`
- Training loss: scalar

The model interface is intended to be:

```python
logits, loss = model(input_ids, targets=None)
```

When `targets` is provided, the model computes next-token loss. When `targets` is absent, it returns logits for generation.

The exact configuration and implementation must be defined in `aurora/model/config.py` and documented in `docs/architecture/`.

---

## 4. Dataset Strategy

AURORA targets code generation and completion. Python is the initial focus because it supports relatively accessible preprocessing, syntax validation, and executable evaluation.

### Initial strategy

1. Start with a tiny synthetic corpus to validate the training mechanics.
2. Evaluate candidate real-world code datasets.
3. Audit access conditions, dataset terms, source licenses, and provenance.
4. Prepare a small Python-heavy pilot dataset.
5. Filter malformed, irrelevant, or unsuitable examples.
6. Deduplicate samples and reduce train/validation contamination.
7. Train or configure a tokenizer.
8. Convert the corpus into token sequences for next-token prediction.

### Dataset candidates to investigate

**CodeSearchNet**  
https://github.com/github/CodeSearchNet

Potentially useful for validating the data pipeline and working with code and natural-language documentation.

**The Stack / The Stack v2**  
https://huggingface.co/datasets/bigcode/the-stack

Potential sources for code training data, subject to access requirements, individual source licenses, dataset terms, and practical feasibility.

These are candidates, not pre-approved training datasets. The data owner must verify the relevant conditions before selecting and downloading a subset.

### Initial data scope

Start with a small pilot, potentially around 100 MB–1 GB of raw data, depending on access and preprocessing constraints. This is a planning estimate, not a proven sufficient pretraining corpus.

Do not download an entire large corpus merely because it is available.

### Tokenization

The initial tokenizer candidate is byte-level BPE (byte-level byte-pair encoding), implemented using an established tokenizer library.

The tokenizer must expose a consistent interface:

```python
tokenizer.encode(text)       # list[int]
tokenizer.decode(token_ids)  # str
tokenizer.vocab_size         # int
```

The model vocabulary size must match the tokenizer exactly. Special tokens, sequence boundaries, padding, and loss masking must be documented.

### Data governance

For every selected source, record:

- Dataset name and version.
- Source URL and access date.
- License and applicable usage conditions.
- Processing and filtering steps.
- Deduplication method.
- Training/validation split method.
- Dataset manifest and content hashes where practical.

---

## 5. Training and Evaluation Strategy

### Training

The training system must support:

- Reproducible configuration and random seeds.
- Forward pass and loss computation.
- Backpropagation and optimizer updates.
- Training and validation metrics.
- Checkpoint saving and resuming.
- GPU memory and throughput measurements.
- Logging and experiment metadata.
- Interrupted-session recovery.

A complete checkpoint should contain model state, optimizer state, training step, relevant scheduler/scaler state, and sufficient configuration metadata to resume the experiment consistently.

### Evaluation

AURORA must distinguish language-model learning from coding capability.

Initial metrics:

- Training loss.
- Held-out validation loss.
- Perplexity where meaningful and computed consistently.
- Tokens processed per second.
- Peak memory consumption.
- Model parameter count.
- Checkpoint size and storage representation.
- Generation latency.
- Syntax validity.
- Code execution or unit-test success on a controlled benchmark, when available.

Code evaluation must be isolated from the host environment. Generated code must not receive unrestricted filesystem, network, credential, or system access.

### Experimental comparisons

When comparing the floating-point and ternary-weight paths:

- Use comparable architecture sizes where possible.
- Use the same tokenizer and dataset split.
- Record training tokens and compute budgets.
- Record software versions and hardware.
- Report failed or unstable experiments.
- Avoid attributing improvements to quantization when other settings differ.

The team must not claim that ternary weights reduce inference latency or memory unless those improvements are measured in the actual implementation.

---

## 6. Repository Structure

```text
aurora/
├── README.md
├── context.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── smoke_test.yaml
│   ├── baseline_small.yaml
│   └── ternary_small.yaml
│
├── docs/
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── tensor_shapes.md
│   │   └── module_contracts.md
│   ├── research/
│   │   ├── bitnet_b1_58_notes.md
│   │   └── design_decisions.md
│   ├── data_policy.md
│   └── experiment_protocol.md
│
├── aurora/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── attention.py
│   │   ├── normalization.py
│   │   ├── feedforward.py
│   │   ├── block.py
│   │   ├── transformer.py
│   │   └── initialization.py
│   │
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ternary.py
│   │   ├── bitlinear.py
│   │   ├── activation_quantization.py
│   │   └── diagnostics.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── dataset.py
│   │   ├── packing.py
│   │   ├── prepare.py
│   │   ├── filtering.py
│   │   └── manifest.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── checkpoint.py
│   │   ├── precision.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── generate.py
│   │   ├── perplexity.py
│   │   ├── code_metrics.py
│   │   ├── benchmark.py
│   │   └── report.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── logging.py
│       └── device.py
│
├── scripts/
│   ├── smoke_test.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── profile.py
│
├── tests/
│   ├── model/
│   ├── quantization/
│   ├── data/
│   ├── training/
│   └── integration/
│
├── data/                  # Local artifacts; do not commit
├── experiments/            # Local run outputs; do not commit
└── reports/
    ├── experiment_template.md
    └── benchmark_template.md
```

This is the intended structure, not a requirement to create every file immediately. Add components as their dependencies become clear.

---

## 7. Module Responsibilities

| Module | Responsibility |
|---|---|
| `model/config.py` | Model dimensions, hyperparameters, and configuration validation |
| `model/attention.py` | Causal self-attention |
| `model/block.py` | Transformer block assembly |
| `model/transformer.py` | End-to-end next-token prediction model |
| `quantization/ternary.py` | Ternary quantization and scaling |
| `quantization/bitlinear.py` | Linear layers using the selected ternary method |
| `data/tokenizer.py` | Encoding and decoding |
| `data/dataset.py` | Training and validation samples |
| `data/prepare.py` | Dataset preparation and provenance |
| `training/trainer.py` | Training and validation loop |
| `training/checkpoint.py` | Saving and resuming experiment state |
| `evaluation/generate.py` | Autoregressive token generation |
| `evaluation/code_metrics.py` | Coding-specific evaluation |
| `scripts/` | Command-line entry points |
| `tests/` | Unit and integration tests |

---

## 8. Team Organization

The project has seven members.

### M1 — Project lead and architecture integration

- Repository structure and shared contracts.
- Configuration and architecture decision records.
- Pull-request reviews and integration.
- Coordination across workstreams.

### M2 — Transformer architecture

- Embeddings, attention, normalization, feed-forward layers, and Transformer blocks.
- Model unit tests and tensor-shape documentation.

### M3 — Ternary quantization

- BitNet b1.58 research.
- Quantization and scaling specification.
- Ternary linear layers, gradient strategy, and numerical tests.

### M4 — Dataset research and governance

- Corpus comparison and access checks.
- License and provenance review.
- Dataset selection, filtering, and deduplication policy.

### M5 — Tokenizer and data pipeline

- Tokenizer implementation.
- Encoding/decoding.
- Sequence packing and dataset interfaces.
- Data pipeline tests.

### M6 — Training infrastructure

- Training loop, optimizer, validation, checkpoints, and logging.
- Compute feasibility and profiling.

### M7 — Testing and evaluation

- Automated test infrastructure.
- Evaluation metrics and reproducibility.
- Generation tests, code evaluation, and experiment reports.

Every module should have one accountable owner. Changes to shared interfaces must be discussed with affected owners before merging.

---

## 9. Development Workflow

Use Git and GitHub for collaborative development.

1. `main` must contain tested, reviewable code.
2. Each member works on a feature branch.
3. Each task has an explicit acceptance criterion.
4. Changes enter `main` through reviewed pull requests.
5. Automated tests should run on every pull request.
6. Data, checkpoints, credentials, and large generated artifacts must not be committed.
7. Each experiment must record its configuration, dataset manifest, tokenizer version, code revision, and results.

Example branches:

```text
main
├── feat/model-attention
├── feat/model-transformer
├── feat/ternary-quantization
├── feat/dataset-pipeline
├── feat/tokenizer
├── feat/training-loop
└── feat/evaluation
```

---

## 10. Current Development Phase

The initial phase is architecture and pipeline development.

### First-week objective

- Establish the repository and interfaces.
- Implement and test causal attention.
- Produce the ternary-weight design specification.
- Audit candidate datasets and select a pilot.
- Prototype the tokenizer and a synthetic training step.
- Set up starter tests and evaluation metrics.

### First-week exit criteria

The team should have:

- A repository with documented conventions.
- An agreed model API and tensor-shape specification.
- A tested causal attention implementation.
- A written ternary quantization and gradient specification.
- A dataset recommendation with supporting evidence.
- A tokenizer prototype.
- A runnable training-step prototype.
- Initial tests and an experiment-report template.

Do not begin long GPU training runs before the first end-to-end smoke test and checkpoint-resume validation.

---

## 11. Compute Constraints

The project is initially compute-constrained.

The team should use local machines for code development, data inspection, CPU tests, and small correctness experiments. Kaggle GPU sessions can be used for controlled training and profiling.

The project must measure actual GPU availability, memory, throughput, and session limits before selecting a model size or training-token budget.

Do not assume that a particular parameter count or dataset size is feasible until it has been benchmarked.

Checkpoints must support recovery across separate sessions.

Paid cloud compute is not assumed in the initial plan. Any future request for institutional GPU resources should be supported by a measured experiment plan, preliminary results, and a bounded compute requirement.

---

## 12. Longer-Term Research Direction

After the standalone model works, AURORA may explore a coding-agent architecture that combines a compact model with stronger external or locally hosted components.

Possible future research directions include:

- Adaptive routing between generation, retrieval, execution, verification, and repair.
- A compact workhorse model with selective escalation to a stronger model.
- Execution-aware correction and repair.
- Distillation of verified corrections into the smaller model.
- Low-bit inference kernels and quantized storage.
- Speculative generation and semantic-level intervention.
- Evaluation of coding capability per unit of latency, memory, or compute.

These are future hypotheses, not current implemented features. The first priority remains a correct, reproducible standalone language model.

---

## 13. Instructions for AI Coding Assistants

Any LLM or coding assistant working on AURORA must follow these rules.

1. Read this document and the relevant module contracts before proposing or changing code.
2. Do not redesign the entire project without an explicit request.
3. Do not invent experimental results, dataset permissions, benchmark scores, or claims of novelty.
4. Do not describe a simple ternary quantizer as a faithful BitNet b1.58 implementation without verifying the method against the paper.
5. Preserve the distinction between the floating-point reference model and the ternary-weight model.
6. Keep module interfaces stable. Discuss breaking changes with the project lead and affected owners.
7. Provide tests with every meaningful implementation.
8. Explain tensor shapes, mathematical operations, numerical assumptions, and non-obvious implementation decisions.
9. Prefer small, verifiable changes over large rewrites.
10. Do not launch expensive training jobs without explicit approval and a defined compute budget.
11. Do not download large datasets before access, licensing, storage, and filtering requirements have been reviewed.
12. Treat code generation and execution as security-sensitive; generated programs must run in a restricted environment.
13. Distinguish documented research findings, project decisions, and unverified hypotheses.
14. When uncertain about a design decision, identify the uncertainty and propose a test rather than silently guessing.

### Expected response format for substantial coding tasks

When implementing a module, the assistant should provide:

- Purpose and design rationale.
- File path.
- Complete code for the requested file.
- Tests and expected behavior.
- Integration instructions.
- Known limitations and unresolved questions.

Do not replace requested code with vague pseudocode unless the user specifically asks for a design or specification.

---

## 14. Definition of Success

AURORA's initial success is a reproducible, technically defensible experiment demonstrating that the team can:

- Implement a decoder-only language model.
- Specify and test ternary-weight behavior.
- Train and resume experiments reliably.
- Curate and document a suitable code dataset.
- Measure model quality and compute requirements.
- Evaluate generated code under controlled conditions.
- Explain limitations and identify evidence-based next steps.

The long-term objective is to determine whether a compute-efficient, ternary-weight coding model can become a useful component of a coding system. That outcome must be established experimentally rather than assumed in advance.

---

**Subject to change anytime required.**
Made by Ankush Tarafdar and GenAI tools