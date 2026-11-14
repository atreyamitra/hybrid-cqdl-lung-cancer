# Changelog

Three layers of work are kept separate here:
**inherited** (on GitHub `main` before this task: February baseline and September upgrade),
**integrated** (the November 2026 refinement, prepared locally earlier and first pushed with this branch), and
**added for the end-semester submission** (this branch, `endsem-finalization`).

## End-semester finalization — branch `endsem-finalization` (added in this task)

### Added
- **Interactive local app** (`app/streamlit_app.py`, logic in `src/app_core.py`, Windows launcher `run_app.bat`).
  - Inference page: choose a saved checkpoint, score a bundled synthetic fixture or an uploaded prepared `.npy` crop, preview the image, and see the model, depth, data kind, sigmoid score, fixed 0.5 threshold and predicted class.
  - Clear rejection of PNG/JPEG/DICOM files, pickled or malformed arrays, wrong shapes, NaNs and 8-bit ranges.
  - Synthetic checkpoints and inputs are labelled, and the score is stated to be uncalibrated. Inference is deterministic (eval mode, `torch.inference_mode()`, gradients disabled); the app never trains.
  - Results page reads saved files: aggregate and per-run tables, confusion matrices, training curves, parameter counts, measured training time with machine context, diagnostics and the original single-seed runs.
- **Matched-bottleneck classical comparator** (`src/matched_classical.py`, `--model matched`). It uses the same convolution blocks and the same Linear(57,600→8), then Linear(8→8)+tanh in place of the quantum layer, then Linear(8→1)+sigmoid. That is 479,705 parameters versus 479,657 for the 3-layer hybrid; matched in width and interface, not exactly in parameter count.
- **Configurable circuit depth** (`--q-layers`, default 3). Depth is saved in checkpoints and rebuilt by every loader. Older checkpoints without a depth field load as three layers.
- **Experiment runner** (`src/experiments.py`). It trains configurations × seeds on one fixed split and writes per-run evidence plus `aggregate.json`/`aggregate.csv`/`runs.csv`. The aggregate uses mean and sample std over defined values, with explicit `n_defined`. Completed runs are never overwritten (`--resume` skips them).
- **Executed experiment** `experiments/endsem_synthetic/`: matched, hybrid L=1, hybrid L=3 and original classical × seeds 42/43/44 on synthetic fixtures (12 runs), plus a quantum-layer diagnostic for each hybrid run.
- Training-loop wall-clock time (`train_seconds`) and hardware/runtime context (`run.json → runtime`) recorded for every run.
- `src/models.py` maps a model name and depth to an architecture.
- 15 tests in `tests/test_endsem.py`, bringing the total to 39.
- `scripts/make_figures.py` now also draws per-seed validation curves.
- `streamlit==1.67.0` added to `requirements.txt`.
- `.github/workflows/ci.yml` also runs prediction, the diagnostic and a one-epoch experiment. **This file is in the submission ZIP but not on the branch**: the GitHub token available for pushing lacks the `workflow` permission, so GitHub refused the change. Copy it from the ZIP to update CI.

### Changed
- `DEMO.html` links to the image files instead of embedding them (`python scripts/build_demo.py --embed` makes the single-file version), and it summarises the new experiment. The Streamlit app is now the primary demonstration.
- The prediction output includes `q_layers`. Plot titles name the matched model and the depth.
- `.gitignore` also excludes experiment checkpoints, plots and per-run split copies, and generated documentation figures. `.gitattributes` keeps CRLF for `.bat` files.
- CSV evidence is written with LF line endings (previously Python's CRLF default); existing CSVs were converted with values unchanged.
- Fixed two unclosed CSV file handles (ResourceWarnings); `run_demo.bat` now points to `START_HERE.md`.

### Unchanged on purpose
- Default architecture, preprocessing, data contract, split, hyperparameters and evaluation rules. Seed-42 runs reproduce the earlier single-seed results exactly.

## November 2026 — refinement (integrated: prepared locally, first pushed with `endsem-finalization`)

A bounded audit-and-fix pass over the September upgrade. The model architecture, preprocessing contract, hyperparameters and checkpoint format are unchanged, and September checkpoints still load.

### Fixed

- **Hidden patient leakage.** A patient ID with a trailing or leading space (common in spreadsheet exports) was treated as a different patient, so the same patient could appear in train and test without an error. All manifest fields are now trimmed, and IDs that differ only by letter case are rejected.
- **Crash on short CSV rows.** Rows with missing fields now raise a `ValueError` that gives the line number instead of an `AttributeError`.
- **Non-portable evidence.** `split_manifest.csv` and `predictions.csv` used to record absolute paths from the training machine. Paths are now relative to the manifest, and `run.json` never stores an absolute path.
- **Split coupled to training seed.** `--seed` also reshuffled the automatic patient split. A separate `--split-seed` (default 42) now controls the split, so future multi-seed runs vary only initialisation and batch order. Default behaviour is unchanged.
- **Prediction CLI robustness.** Missing files, non-`.npy` inputs, pickled arrays, bare state dicts and mismatched weights now produce one-line errors with exit code 2 instead of tracebacks. The loader and predictor share a single crop validator.
- **Wording of scores.** The prediction output field is now `score_class_1`, alongside `calibrated_probability: false`, because a sigmoid output from a synthetic-trained model is not a calibrated probability. (It was previously `probability_class_1`.)
- Training aborts on a non-finite validation loss instead of failing later without a checkpoint.

### Added

- `src/diagnose.py`: measures the spread of encoded angles and Pauli-Z expectation values across a manifest. It explains why the hybrid's scores cluster at 0.506–0.515.
- 12 tests in `tests/test_robustness.py`, bringing the total to 24.
- `scripts/make_figures.py` and `scripts/build_demo.py`, which regenerate every report/slide figure and `DEMO.html` from saved evidence.
- `results/archive_2026-09/`, which keeps the September evidence including the six plots recovered from the old demo page.
- `results/reproduction_check.json`, which compares the September and November runs (identical metrics).
- Plots now mark the selected epoch and show the sample size. The ROC title no longer clips.
- The CI workflow also runs the prediction and diagnostic CLIs.
- `run_demo.bat` prefers `py -3.12`, falls back to PyPI torch, keeps the shipped fixtures and uses the shipped checkpoint when present.

### Documentation

- README, audit record, dataset protocol, START_HERE, verification summary and continuation handoff are rewritten to match the code.
- The report, viva guide and presentation are rebuilt (the earlier report and viva DOCX files were missing from the supplied archive). The presentation is now editable, names both students, uses code-matched diagrams and fixes the overlapping titles.
- The repository state was corrected: the September code is on GitHub (commit e345a73).

## September 2026 — Semester 5 upgrade (inherited: GitHub commit e345a73)

- Validated CSV manifests of prepared crops, with patient-separated splits and checks for duplicates, shape and range.
- Seeded CPU training for both models, validation-loss checkpoint selection, fixed 0.5 threshold and held-out test evaluation.
- Saved evidence (checkpoint, metrics, history, configuration, predictions, plots) and an inference CLI with weights-only loading.
- 12 unit tests, 24 synthetic fixtures and two five-epoch synthetic runs.
- Offline demo page, Windows launcher and GitHub Actions workflow.

## February 2026 — original baseline (inherited)

- Four files: `classical_cnn.py`, `hybrid_model.py`, `quantum_layer.py`, `train.py` (random tensors, random labels). They are preserved in `baseline_src/`.
