# Paper 2: Prediction-Constructed Relations and Error Locking

This repository contains the analysis code, frozen protocols, and result artifacts for the paper on prediction-constructed hypergraph relations, error locking, and correction limits.

## Repository map

- `src/analysis/`: experiment and analysis scripts for the matched-error controls, prediction-based tests, theoretical checks, and supplementary analyses.
- `src/upgrade/`: the frozen controlled-correction and near-locking code used by the manuscript.
- `src/official_relation_audit/`: observer and audit scripts for the pinned official IDGL and Pro-GNN implementations.
- `results/analysis/`: saved outputs from the analysis scripts, including the detailed sign-prediction records and plots.
- `results/controlled_correction/`: frozen protocol, formal outer-test output, node diagnostics, and the decision log.
- `results/official_relation_audit/`: aggregate and per-node outputs for the two descriptive official-code audits.
- `results/mechanism-reanalysis.json`: saved-trajectory reanalysis and its input hashes.
- `data/README.md`: input-data provenance and non-redistribution notes.

## Reproducing the released analyses

Python 3.10+ is recommended. The NumPy/SciPy analysis scripts need `numpy` and `scipy`; plotting scripts additionally need `matplotlib`; predictor scripts may need `scikit-learn`. The official audit wrappers require the dependencies of the pinned upstream repositories. GPU training is not needed to inspect or recompute the released aggregate and per-node audit tables.

The archived scripts were developed in a research workspace and expect dataset files under the relative `data/` directory. Dataset files are intentionally not bundled. Obtain the inputs from their original sources and place them as described in `data/README.md`. This release does not claim a clean, one-command reproduction of every historical exploratory run.

The correction protocol and result are in `results/controlled_correction/protocol.json` and `results/controlled_correction/correction_validation.json`. Deterministic protocol tests are provided in `src/upgrade/test_upgrade.py`. The official relation audit outputs are in `results/official_relation_audit/`; their exact upstream repository commits and audit limits are recorded alongside the outputs.

## Reading the results

The current manuscript distinguishes exact mathematical locking results, controlled synthetic/matched-error evidence, prediction-based dataset experiments, and descriptive single-checkpoint audits. Pilot runs, alternative parameterizations, and exploratory results in `results/analysis/` are not all manuscript headline results. Read each result's protocol and the manuscript's stated scope before interpreting it. The official IDGL and Pro-GNN audits use one prespecified checkpoint per method and are not estimates over training randomness.

No trained model weights, raw benchmark datasets, or full captured relation matrices are redistributed here. Node-level audit summaries and analysis outputs are included; the upstream official implementations and datasets remain subject to their own terms and licenses.

## License and citation

The original experiment and analysis scripts in this repository are released under the MIT License (see `LICENSE`). Third-party software and datasets are not relicensed by this notice. Please cite the associated manuscript when using these materials; the formal citation and versioned DOI will be added when available.
