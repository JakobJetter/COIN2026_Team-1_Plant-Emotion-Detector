# Re-analysis for the Sensors manuscript (Gloor et al., 2026)

Scripts added for the journal version of this project, Study 2 of
*Plant Bioelectric Potentials as a Proxy for Human Emotional Valence*.
They run on `data/datasets/consolidated_dataset.csv` from the repository root
(`python analysis_sensors/<script>.py`) and reproduce the numbers in
Sections 4.8.1–4.8.5 of the manuscript.

| File | Purpose | Manuscript |
|---|---|---|
| `study2_rerun_14participants.py` | Full re-run of the three model pipelines (RF/XGBoost/MLP × raw/session-normalised, nested participant-grouped CV, 200 label permutations), the time-only baseline, the time-preserving permutation test and the CO₂ partial-correlation analysis on the 14-participant dataset | Tables 3–5, Figure 10 |
| `study2_14participants_results.json` | Output of the script above | Tables 3–5 |
| `time_baseline.py` | Stand-alone time-only baseline (window index / elapsed time as the only features) | Table 4 |
| `time_vs_plant.py` | Plant-only vs. time-only vs. joint models, detrended features, and the time-preserving permutation null for valence | Section 4.8.4 |
| `figure10_classifier_comparison_14participants.png` | Classifier comparison figure | Figure 10 |

The dataset in `data/datasets/consolidated_dataset.csv` contains the 14 participants
(265 windows) analysed in the manuscript. The session of one participant who could not be
reached for retrospective written consent to publication (participant 09 of the seminar
recording) was removed from the public dataset; the seminar report (`project_report.pdf`)
still describes the original 15-participant analysis.
