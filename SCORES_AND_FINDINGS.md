# Scores And Findings Summary

## Best Reported Scores

| Source | Setting | Accuracy | F1 | ROC-AUC | ECE | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `docs/final_report.md` | Multi-seed hybrid QCNN | 0.8842 | 0.8791 | not listed | 0.0421 | Best calibrated research-style result found in the docs |
| `docs/runner_fix_report.md` | Post-fix evaluation run | 0.8790 | 0.8790 | 0.9680 | not listed | Shows convergence fix improved metrics |
| `README.md` | Production-style hybrid result | 0.7940 | 0.7940 | 0.9300 | not listed | Canonical README headline result |
| `docs/analysis_report.md` | Transformer baseline ceiling | 0.8790 | not listed | not listed | not listed | Reported as the top benchmark family |
| `evaluation/latest/global_benchmark/results_N50000_latest.csv` | Classical baseline | 0.5822 | 0.58145 | not listed | 0.02062 | Baseline comparison |

## Lower-Stability Or Stress-Test Results

| Source | Setting | Accuracy | F1 | ROC-AUC | ECE | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `evaluation/latest/metrics/cross_language_results.json` | Zero-shot transfer | 0.31792 | 0.15338 | 0.45778 | 0.13915 | Holdout generalization is much weaker |
| `evaluation/latest/metrics/language_metrics.json` | English stage | 0.77640 | 0.77535 | 0.91577 | 0.03236 | Strong language-specific run |
| `evaluation/latest/metrics/language_metrics.json` | Hindi stage | 0.42460 | 0.41533 | 0.58684 | 0.21134 | Much weaker than English |
| `evaluation/latest/metrics/language_metrics.json` | Bhojpuri stage | 0.55646 | 0.54746 | 0.74733 | 0.05555 | Moderate performance |
| `evaluation/latest/metrics/quantum_gain.json` | Hybrid vs baseline comparison | 0.59293 | 0.0 | not listed | not listed | Slightly negative gain in this snapshot |

## Main Findings

1. The repo has strong calibrated runs, but not one universal score.
2. The best numbers live around 0.88 accuracy and 0.88 F1.
3. Production-style and transfer-style results are not the same experiment class.
4. The quantum layer is not the only determinant of performance; data split, language, and calibration all matter.
5. The current workstation is capable enough to run the project, but the QCNN simulation still benefits from careful batch sizing and a correct Python environment.

## Bottom Line

If you need one headline for the project, use this:

The repository’s strongest validated hybrid QCNN result is approximately 0.884 accuracy with 0.879 F1 and 0.042 ECE, while the strongest production-style README result is 79.4% accuracy with 0.930 ROC-AUC. Holdout transfer is much weaker and should be treated as a stress test, not the core benchmark.
