# Project audit record

## Part 1 — September 2026: baseline audit and bounded continuation

Source: supplied GitHub ZIP and complete archived report/PPT, inspected September 2026. No use of UPDATED_PBL_REPORT.docx.

| Baseline weakness | Resolution or current boundary |
|---|---|
| TensorFlow/Keras prose disagreed with code | PyTorch + PennyLane throughout current materials |
| Dummy random-image training | Explicit geometric demo plus prepared-crop training interface |
| No patient-level data identity | Required patient ID, split isolation checks |
| No duplicate or shape validation | Validated arrays and exact content hashes |
| No held-out protocol | Separate train/validation/test with class coverage checks |
| Unverified historical accuracy | Archived as unverified, replaced by labeled current smoke-test results |
| QCNN/64-qubit/amplitude/pooling descriptions | Eight-qubit variational layer, RX angle encoding, no quantum pooling |
| RY/parameter-shift claims | Explicit RX gates and analytic simulator backpropagation |
| No saved experiment evidence | Checkpoints, config, hashes, predictions, metrics and plots |
| Unsupported efficiency/overfitting claims | Measured parameter counts with bottleneck confounding explained |

## Semester scope

The requested approximate 30% upgrade is a scope constraint, not a measured improvement percentage. Three additions: validated data handling, reproducible experiments and a demonstrable evidence package. The CNN, eight-feature bottleneck and 3×8 quantum weights are preserved. Do not describe this as a new model architecture or 30% better accuracy.

The archived report's membership includes Atreya Mitra and Debraj Mondal. The old deck foregrounds Atreya. Current report retains the original team record; no new per-person work allocation is invented. Supervisor details are retained from the supplied source. The certificate page is unsigned and makes no claim of fresh faculty approval.

## Remaining external work

Acquire and prepare real patient crops and verified labels. Real-data training then uses the delivered CLI without rewriting model code. Clinical metrics, dataset counts and medical claims remain pending until that run exists. The Semester 5 source is synchronized to the GitHub repository. Generated checkpoints, bulk synthetic arrays and result plot binaries remain in the submission archive rather than Git history. No hardware account or hardware experiment is needed for this semester's scoped implementation.

## Part 2 — November 2026 refinement audit

Scope: verify and refine the September upgrade, not a second 30% expansion. Sources inspected: the GitHub repository (commit e345a73), the surviving entries of the supplied submission ZIP (truncated in transfer — report, viva guide, START_HERE, src/ and tests/ were missing and one checkpoint was corrupt), two edited 19/18-slide decks (image-only slides) and the September handoff note.

| Finding | Evidence | Action |
|---|---|---|
| Patient IDs with stray spaces (e.g. "P01 ") counted as different patients, so leakage across splits went undetected | executed probe: manifest with `synthetic_01 ` in test and `synthetic_01` in train was accepted | IDs, labels, splits and paths are trimmed; case-only variants rejected; regression test added |
| Short CSV rows crashed with `AttributeError` | executed probe | row-numbered `ValueError` |
| Evidence CSVs recorded absolute local paths; committed September files had been hand-edited to relative paths | code inspection + re-run output | paths written relative to the manifest; absolute paths never written; test added |
| `--seed` also changed the automatic patient split, confounding future multi-seed runs | code inspection | new `--split-seed` (default 42, so defaults unchanged) |
| Prediction CLI showed raw tracebacks for missing files, PNGs and bare state dicts, and called the sigmoid output a probability | executed probes | shared crop validation, clear errors (exit 2), output field renamed `score_class_1` with `calibrated_probability: false` |
| Hybrid scores all ≈0.51 were described but not explained | measured with new `src.diagnose` | expectation-value spread 0.014 and classifier weights 0.11 documented as the cause in this run |
| September runtime ratio (35.8% lower hybrid time) presented as a headline | re-run gave a different ratio | treated as an observation only |
| Deck: Debraj Mondal missing from title; titles overlapping subtitles; an AI-generated architecture picture labelled the input "CT crop" and the output "malignancy probability" | visual inspection of every slide | deck rebuilt as editable slides with code-matched diagrams |
| Old handoff said the code was not pushed to GitHub | GitHub shows commit e345a73 on 15 Sep 2026 | corrected; November changes are not pushed |

Re-running both September experiments with the November code reproduced every metric, confusion matrix and selected epoch exactly (`results/reproduction_check.json`).

## Part 3 — End-semester finalization (branch `endsem-finalization`)

Reconciliation: GitHub `main` was still commit e345a73 (one branch, no pull requests), so nothing on GitHub was newer than the local November package. The November refinement was therefore applied on top of `main` unchanged, and the end-semester work was added after it.

| Requirement | Implementation | Evidence |
|---|---|---|
| Interactive local inference | Streamlit app with inference and results pages; inference-only, deterministic, rejects non-crop inputs | `tests/test_endsem.py` (input cases, determinism, both pages rendered with Streamlit's AppTest); headless launch with health check `ok`; screenshots |
| Matched comparator | Linear(8,8)+tanh replaces the circuit; 479,705 parameters (3-layer hybrid: 479,657) | unit test of shapes and counts |
| Depth experiment | `--q-layers`, saved in checkpoints; legacy checkpoints load as L=3 | round-trip and legacy tests |
| Three-seed experiment | `src.experiments`; 12 runs executed on one fixed split | `experiments/endsem_synthetic/` |
| Results view | App results page reads saved JSON/CSV/PNG only | AppTest render |

Seed 42 of the 3-layer hybrid and of the original classical model reproduce `results/hybrid` and `results/classical` exactly, so adding depth and the matched model did not change the original configurations.
