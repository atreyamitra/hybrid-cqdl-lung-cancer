# Continuation handoff — state at end of Semester 5 (November 2026)

Project: Hybrid Classical–Quantum Deep Learning for Lung Cancer Detection. Manipal University Jaipur, 2026–27. Atreya Mitra (2427030002), Debraj Mondal (2427030392), Section H. Guide: Mr Tapan Kumar Dey. Continues next semester as a three-credit project.

## Where the work is

- GitHub `atreyamitra/hybrid-cqdl-lung-cancer`: `main` holds the September 2026 state (commit e345a73). Branch `endsem-finalization` holds the November refinement and the end-semester additions, with an open pull request (not merged).
- The submission ZIP holds the same code plus the binaries git does not store: checkpoints, plots and fixtures.

## What exists (verified by execution on Linux CPU)

- **Models.** Hybrid CNN + 8-qubit variational layer (RX encoding, *L* BasicEntanglerLayers, default *L* = 3, 8 Pauli-Z outputs). Original classical CNN (128-unit head). Matched-bottleneck classical CNN (Linear(8,8)+tanh in place of the circuit).
- **Data.** Validated manifest loader for prepared 128×128 `.npy` crops, with patient-leakage, duplicate, shape and range checks; 24 synthetic fixtures.
- **Protocol.** Seeded training, validation-loss checkpoint selection, fixed threshold 0.5, held-out test, full evidence per run, with depth and runtime recorded.
- **Tools.** `src.train`, `src.experiments`, `src.predict`, `src.diagnose`, a Streamlit app (inference + results) and figure/demo builders.
- **Evidence.** `results/` (original single-seed runs), `results/archive_2026-09/`, and `experiments/endsem_synthetic/` (4 configurations × 3 seeds, aggregate and diagnostics).
- **Tests.** 39 passing.

## What the evidence says

Synthetic data only, with 4 test images per run. Every configuration has AUC 1.0, while thresholded metrics vary strongly with the seed. The 3-layer hybrid predicts all test images positive under every seed; its quantum outputs barely vary between images (spread 0.008–0.024). No model ranking, clinical claim or quantum-advantage claim is supported.

## Next semester, in order

1. **Real data.** Build a LIDC-IDRI pipeline following `docs/DATASET_PROTOCOL.md` (DICOM → resampled, nodule-centred 128×128 crops), with a label rule reviewed by the guide and reviewed patient-level splits.
2. Run the existing experiment runner on the real manifest, using all four configurations and at least 5 seeds, with a test set large enough for confidence intervals (bootstrap).
3. Hybrid trainability: angle scaling before encoding, initialisation, learning rate, longer training. Measure each change with `src.diagnose`. This is a small, planned study, not a broad hyperparameter search.
4. Grad-CAM on the convolution blocks.
5. Quantum noise simulation, then external validation.

Longer term: quantum hardware, federated and multimodal learning.

## Rules

Do not change defaults without regenerating evidence. Write new runs to new folders and never tune on the test split. Label every result with its data kind. Keep `results/archive_2026-09/` and `experiments/endsem_synthetic/` as historical evidence.

## Not verified / not pushed

The Windows launchers (`run_app.bat`, `run_demo.bat`) and the GitHub Actions workflow have never been executed. The updated `.github/workflows/ci.yml` is in the submission ZIP but not on the branch, because the token used for pushing lacks GitHub's `workflow` permission. Commit it from a local clone to update CI.
