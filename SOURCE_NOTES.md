# Implementation notes

REHSurv provides local training and held-out testing with no SSH dependency.
The main.py entry exposes the main architecture and four manuscript cohorts.
Patient folds remain fixed; validation C-index selects checkpoints.

The requested time-bin count (8), LR plateau patience (3), minimum C-index
improvement (0.001), and gradient clipping (5) are implementation settings
retained from the training code; their numerical values are not all specified
in the manuscript. Input UNI2-h features and 186 KEGG pathway definitions
must be supplied separately.

Regression corrections include coordinated-negation handling, validation
comparability preflight and runtime guards, case-insensitive feature suffixes,
and checkpoint data-directory reuse unless explicitly overridden.
No model hyperparameters were changed by these corrections.

The English rule extractor is heuristic. Regenerate assertion caches and
retrain models to apply changes to extraction. Synthetic tests verify software
behavior; they do not reproduce real-cohort manuscript metrics.
