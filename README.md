# Multi-Aspect Review Scoring: LoRA and Reliable Evaluation

Predict appearance, aroma, palate, taste and overall ratings from review text.

**Team project led by Ashley Liu (Xinying Liu).** Ashley reports completing most of the implementation. Exact module-by-module teammate attribution remains to be documented. The original DistilBERT + LoRA experiment is preserved alongside an AI-assisted reliability extension with runnable baselines, product-held-out evaluation and checkpoint auditing.

## Try it locally

Python 3.10+ and NumPy are sufficient. No API key, model download or private dataset is needed for the synthetic demo.

```bash
python -m pip install -r requirements.txt
python -m reviewscore predict --model examples/synthetic-baseline.json --text "Bright floral aroma with a smooth finish"
python -m reviewscore benchmark --data examples/synthetic.jsonl --out runs/demo --limit 200
python -m unittest discover -s tests -v
```

**The bundled model is a ridge baseline trained on artificial, templated examples. It demonstrates the workflow, not real-world predictive performance.** Scores are normalized estimates, not confidence probabilities.

## What makes this project useful

- **Partial labels:** invalid or missing ratings are excluded independently for each task rather than converted to genuine zero scores.
- **Unseen-product evaluation:** deterministic 80/10/10 hash partitions keep each product in one partition; normalized exact review duplicates are removed before splitting.
- **Reproducible baseline:** stable signed bag-of-words hashing, ridge regression and train-mean comparison; regularization selected on validation only.
- **Uncertainty:** a paired product-cluster bootstrap measures uncertainty in the test MSE difference, conditional on the fitted models.
- **Recoverable serving artifacts:** the baseline uses JSON and verifies prediction equivalence after reload. The optional neural extension explicitly saves the regression head alongside the LoRA adapter and tokenizer.
- **Checkpoint forensics:** safetensors header inspection identifies incomplete artifacts without loading pickle files.

## Newly executed experiment

A local prefix of 20,000 source records was cleaned and deduplicated. It yielded 15,054 training, 1,310 validation and 3,559 test reviews. The test reviews represent 76 products absent from training and validation. All five target labels were present in these test rows.

| Model | Test macro MSE |
|---|---:|
| Training-set mean for each target | 0.018400 |
| Hashed bag-of-words ridge, 128 features | **0.016149** |

The absolute difference is −0.002250; the exploratory 95% product-cluster bootstrap interval is [−0.003865, −0.001470] using 300 resamples. This is about 12.2% lower MSE than the train-mean baseline **on this prefix sample only**. The sample is not representative of the full corpus, the interval does not include training uncertainty, and this is not a new LoRA result.

[Full metrics and environment](results/product-holdout/metrics.json) · [Technical report](docs/REPORT.md)

## Historical LoRA evidence and limitations

The original implementation adapts DistilBERT attention query/value projections with rank 16 LoRA and predicts five ratings with an MLP. Historical result text reports per-target MSE values, but those numbers are not tied to a complete reproducible checkpoint and split manifest. They must not be compared directly with the new product-held-out experiment.

The checkpoint at step 12,000 records **0.1824 epochs**, has 24 adapter tensors, and contains **no regression-head tensors**. Its configuration has `modules_to_save: null`. The original wrapper also does not explicitly preserve head trainability when applying PEFT; behavior must be checked in the original runtime. The old adapter alone cannot recover the original scorer.

[Checkpoint audit](results/historical/checkpoint_audit.json) · [Historical result text](results/historical/reported_mse.txt)

## Run against your local data

Input is one JSON object or Python-literal dictionary per line, with `review/text`, `beer/beerId`, and optional `review/{aspect}` labels. Fractions such as `13/20` are normalized; plain numeric inputs must already be in [0,1]. Large JSON arrays are deliberately unsupported. The reader uses `ast.literal_eval`, never `eval`.

```bash
python -m reviewscore benchmark --data /path/to/ratebeer.json --limit 20000 --split product --out runs/product-holdout
```

The original data, user names, review texts and individual predictions are not bundled. Acquire data independently under its applicable terms. `--split text` is available as a review-level comparison, but does not hold products out. The benchmark exports aggregate metrics and a local model; output directories must be new or empty.

## Optional neural extension

The new model keeps its trainable regression head outside the PEFT-wrapped encoder, applies an observed-label loss and exports both head and adapter. Validation chooses an epoch; test is evaluated after selection. A reload check compares predictions before reporting success.

```bash
python -m pip install -r requirements-neural.txt
python -m reviewscore.train_neural --data /path/to/ratebeer.json --base distilbert-base-uncased --out runs/lora --limit 20000 --epochs 1
python -m reviewscore.predict_neural --bundle runs/lora/bundle --text "Floral aroma and a balanced finish"
```

A Hub model ID may download model files. Use a local base directory plus `--local-only` for offline execution. CUDA/MPS is opt-in via `--device`; CPU is the default. Dependency ranges are not an exact historical lockfile.

## Verified LoRA smoke run (2026-10-02)

All **13 tests passed**, including masked loss, head trainability and offline bundle roundtrip. A new CPU LoRA run trained for one epoch on the first 2,000 source records: 1,693 training, 260 validation and 46 test reviews after removing one duplicate. Validation macro MSE was **0.026721** and test macro MSE **0.012316**. The exported serving bundle reloaded with matching predictions, and the inference CLI produced five scores.

This small run verifies the end-to-end workflow. Its 46-row test set is too small for a strong generalization claim, and it is not directly comparable with the separate 20,000-record ridge experiment. Original historical weights remain incomplete. The newly trained bundle stays under ignored `runs/`; only aggregate metrics are committed. See [smoke metrics](results/lora-smoke/metrics.json).

The old scripts under `legacy/` are preserved evidence; supported entry points are in `reviewscore/`.


## Repository map

```text
reviewscore/       Data contract, baselines, metrics, CLI, optional neural pipeline
examples/          Synthetic demo data and synthetic baseline model
results/           Aggregate new results and historical evidence
legacy/            Original model and baseline source
tests/            Data, leakage, serialization and optional neural checks
docs/              Report, validation, authorship and source hashes
```

[Validation](docs/VALIDATION.md) · [Attribution](docs/ATTRIBUTION.md) · [Chinese handoff](docs/HANDOFF_ZH.md)
