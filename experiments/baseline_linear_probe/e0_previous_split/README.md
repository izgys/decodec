# E0: first linear probe (historical)

The first baseline, kept for reference. It is not comparable with any current result.

- **Dataset:** the previous split (216,962 sequences, 59 sub-classes), before the cluster-count class filter.
- **Probe:** CLS token, sequences cut at 512 residues, no feature scaling, no class weights, `C=1.0` not tuned.
- **Result:** test macro-F1 0.494, micro-F1 0.679 (single run, no uncertainty estimate).
- **Reproducibility:** `results.json` was produced by an earlier version of `run_baseline.py` than the one stored here.

The corrected probe on the current dataset is `../run_probe.py`.
