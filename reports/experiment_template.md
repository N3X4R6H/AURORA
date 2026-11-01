# Experiment Record: [ID / short name]

**Date:** YYYY-MM-DD
**Author:** Amogh Chhuneja
**Status:** Draft | Running | Completed | Failed

## 1. Question & Hypothesis
- **Question this run answers:** [one sentence]
- **Single variable changed vs. baseline:** [e.g., ternary BitLinear on FFN only]
- **Held constant:** [tokenizer, split, seed, model size, token budget]
- **Expected outcome / decision test:** [what result means adopt, what means reject]

## 2. Reproducibility
- **Git commit:** [hash] (clean working tree? Y/N)
- **Config file:** [configs/....yaml]
- **Dataset manifest / hash:** [path]
- **Train/val split method:** [ ]
- **Tokenizer (name, vocab size, version):** [ ]
- **Seed(s):** [ ]
- **Software versions:** Python / PyTorch / CUDA: [ ]
- **Precision policy:** [weights / activations / master weights / embeddings / head]
- **Checkpoint path & resume tested:** [Y/N]

## 3. Compute Profile
- **Hardware:** [e.g., Kaggle 1x T4, local CPU]
- **Budget (planned):** [GPU-hours / tokens]
- **Actual:** [GPU-hours / tokens / wall-clock]
- **Model:** [params, n_layer, n_embd, n_head, context length]

## 4. Results
| Metric | Baseline | This run |
|---|---|---|
| Train loss (final) | | |
| Val loss | | |
| Perplexity | | |
| Tokens/sec (post-warmup) | | |
| Peak memory: allocated / reserved (MB) | | |
| Checkpoint size (MB) | | |
| Syntax validity / unit-test pass (if run) | | |

## 5. Observations
- **Instabilities, NaNs, divergence, failed attempts:** [report them]
- **Plots/logs:** [path]

## 6. Conclusion
- **Hypothesis supported?** Yes | No | Inconclusive
- **Claims NOT supported by this run:** [e.g., "no inference speedup measured"]
- **Next action:** Adopt | Revert | Refine | Document failure
