# Validation record

Local checks on 2026-10-02, Python 3.12.14 / NumPy 2.3.5:

- 10 tests passed: safe parsing, missing-label semantics, product/text overlap, deterministic splits, serialization, masked target fitting, metric validation and bootstrap direction.
- Subsequently all 13 tests passed in the configured project environment, including all 3 neural tests (user-provided test output).
- Real-data prefix benchmark completed: 20,000 input records, 3,559 held-out test rows. Serving JSON reload predictions matched.
- Synthetic CLI benchmark and inference completed. This fixture is deliberately trivial and is not model-quality evidence.
- Python sources syntax checked. Original source hashes retained.

A subsequent 2,000-record CPU smoke run completed new neural training, bundle reload consistency and CLI prediction. Validation macro MSE: 0.026721; test macro MSE: 0.012316 on 46 rows. Not validated: accelerator execution, full corpus results, original experiment reproduction, or hosted GitHub Actions. NumPy-only checks do not validate PEFT/Transformers compatibility. The repository contains a CI workflow but it has not been run on GitHub during preparation.
