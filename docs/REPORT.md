# Technical report

## Problem and original approach

Multi-aspect scoring maps a review to five normalized continuous values. The source project used a shared DistilBERT representation, an MLP head and LoRA on attention query/value projections. Its comparison script used TF-IDF plus Ridge. This repository retains those sources rather than silently rewriting the original evidence.

## Failure analysis

The source parsers map missing and invalid ratings to zero, conflating absence with a real rating. Random record-level splitting also permits the same product on both sides. The saved adapter does not include the MLP head, preventing exact prediction recovery. A historical text file lists MSE values but lacks a source checkpoint/split identifier; no causal performance claim is made from that file.

## Reliability extension

The new input contract accepts finite normalized scores or fractions with positive denominators. Each target has its own observed-label mask. Exact reviews are deduplicated after lowercasing and whitespace normalization; the first eligible occurrence is retained, so conflicting duplicate records are not merged. Product IDs are used only for splitting and uncertainty estimation, not as input features. Product grouping does not eliminate reviewer overlap, near duplicates, brewery overlap, or all distribution leakage.

Hash buckets provide stable approximate 80/10/10 product assignment. They do not enforce balanced row counts. Ridge features use a deterministic signed hashing map and per-review L2 normalization. The intercept is unpenalized. Each target fits only observed labels; alpha is selected from 0.1, 1 and 10 by validation macro MSE. Test labels never select alpha. Macro MSE gives equal weight to tasks with observed test labels; denominators are reported.

A product-cluster bootstrap resamples products with replacement and includes all their review rows. This respects within-product correlation better than treating every review as independent. The 300-resample percentile interval is exploratory and conditional on the single training run; it is neither a guarantee nor a correction for prefix sampling bias.

## Executed experiment

The first 20,000 records of the local line-oriented source were read; 47 invalid rows and 30 duplicate reviews were excluded. There were 630/72/76 products in training/validation/test. Validation selected alpha 10 for 128 hashed features. Test macro MSE was 0.016149 versus 0.018400 for the training-mean baseline, on 3,559 reviews. Reloaded JSON model predictions matched the original predictions. Exact target-wise MSE/MAE, selected hyperparameters, row digest, environment and run duration are recorded in results/product-holdout/metrics.json.

This sample uses source order, not random population sampling. Do not compare these values against old randomly split TF-IDF/LoRA numbers. The source file is not distributed, so independent reproduction requires obtaining the same input and matching its first-record digest.

## Neural extension design

The encoder alone is PEFT wrapped, while the sigmoid MLP head remains trainable outside it. The masked loss averages observed per-task MSE within each batch. This avoids missing-label penalties but minibatch task weighting is not identical to full-dataset weighting. Validation selects an epoch before the test evaluation. Adapter, head tensors, tokenizer, base reference, revision when available, target order and maximum length form the serving bundle. This is an inference bundle, not an optimizer-resume checkpoint. A historical adapter missing its head is deliberately insufficient.

A subsequent user-run CPU smoke experiment completed training, serving-bundle reload validation and inference; all 13 tests passed. Aggregate metrics are in results/lora-smoke/metrics.json. Full-corpus results and accelerator behavior remain unverified. See the primary PEFT reference on adapters and additional trainable/saved modules: https://huggingface.co/docs/peft/package_reference/lora

## Next substantive experiments

1. Run the new LoRA pipeline against exactly the same partitions and compare it with ridge using paired product-cluster intervals.
2. Sample across the full corpus rather than its prefix, then preregister reviewer-held-out and chronological stress tests.
3. Ablate LoRA rank 4/8/16, attention targets and sequence length using validation only, then evaluate the chosen configuration once on an untouched test split.
4. Add error slices for review length and product frequency, report per-target calibration of continuous predictions and deployment latency. Scores must not be called confidence probabilities.

These are future experiments, not completed features or results.
