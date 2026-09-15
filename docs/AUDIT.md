# Baseline audit and bounded continuation

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
