# Hybrid Classical–Quantum Deep Learning for Lung Cancer Detection

Semester 5 continuation, September 2026. PyTorch + PennyLane, eight qubits, CPU simulation.

This release preserves the existing CNN and variational circuit and adds a validated crop manifest, patient-separated splits, reproducible training, held-out evaluation and a saved demo. It is a research prototype for binary prepared-crop classification. It does not locate nodules in complete scans or provide a clinical diagnosis.

## Start without installation

Open `DEMO.html` to inspect the saved experiment, plots and example prediction. All assets are embedded. The page shows saved outputs; it does not perform live inference.

The submission bundle includes synthetic-data checkpoints for both models. The GitHub repository intentionally keeps generated checkpoints, bulk `.npy` fixtures and result images out of Git history; they can be regenerated from the source commands below. `run_demo.bat` installs dependencies, runs tests, regenerates synthetic data, trains a one-epoch hybrid smoke-test checkpoint, executes one prediction and opens the saved demo on Windows. Initial setup requires internet and Python 3.12. The Windows launcher is supplied but was not executed on Windows; the underlying Python commands were tested on Linux CPU.

## Installation and commands

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.demo_data --out demo_data
python -m src.train --manifest demo_data/manifest.csv --out runs/demo_hybrid --model hybrid --epochs 1 --data-kind synthetic
python -m src.predict --checkpoint runs/demo_hybrid/best.pt --image demo_data/fixture_23.npy
```

Reproduce both demo experiments in new directories:

```bash
python -m src.demo_data --out demo_data
python -m src.train --manifest demo_data/manifest.csv --out runs/hybrid --model hybrid --epochs 5 --data-kind synthetic
python -m src.train --manifest demo_data/manifest.csv --out runs/classical --model classical --epochs 5 --data-kind synthetic
```

The trainer refuses to overwrite a populated run directory. Change the output path for each experiment.

## Architecture

128×128 grayscale crop → Conv(1,32,3) → ReLU → MaxPool(2) → Conv(32,64,3) → ReLU → MaxPool(2) → flatten 57,600 values → Linear(57,600,8) → AngleEmbedding with RX → three BasicEntanglerLayers with RX and CNOT rings → eight Pauli-Z expectations → Linear(8,1) → sigmoid.

`default.qubit` uses analytic expectations and explicit simulator backpropagation. The circuit has 24 trainable rotation angles. It has no quantum pooling, so “variational quantum layer” is more precise than a full QCNN. The classical comparator retains the original dense layer with 128 units. No dropout or L2 was added or claimed.

## Measured evidence

12 unit tests passed. Tests cover split leakage, duplicate content, invalid shapes, class coverage, binary metrics, reproducible splitting, model gradients, parameter counts and checkpoint round trips.

| Model | Parameters | Demo accuracy | Demo F1 | Demo AUC |
|---|---:|---:|---:|---:|
| Classical CNN | 7,391,873 | 0.75 | 0.667 | 1.00 |
| Hybrid | 479,657 | 0.50 | 0.667 | 1.00 |

These are geometric synthetic fixtures: 24 images with distinct mock patient IDs, split 16/4/4. Five training epochs, seed 42, Adam 1e-4, batch size 4. Only four test examples: these figures establish no medical performance or model superiority. AUC measures ranking, so perfect AUC can coexist with poor accuracy at the fixed 0.5 threshold. The original architectures already had different parameter counts; that difference is not a new improvement and is confounded by the 128-versus-8 dense bottlenecks.

The submission archive's `results/{model}/` directories contain `best.pt`, `metrics.json`, `run.json`, `history.json`, `split_manifest.csv`, `predictions.csv`, confusion matrix, ROC curve and training curves. This GitHub repository tracks the compact JSON/CSV evidence while excluding generated checkpoints and plot binaries; `DEMO.html` preserves the saved visual evidence. Runtime in metrics is a local observation including training/checkpoint/evaluation overhead, not a controlled speed benchmark.

## Prepared CT data

See `docs/DATASET_PROTOCOL.md`. Provide one normalized float array per prepared nodule crop, with dimensions 128×128 and values in [0,1]. Provide `path,patient_id,label` in CSV, optionally `split`. Paths are relative to the manifest. Labels must be 0 or 1 with a documented clinical or annotation-derived meaning.

```bash
python -m src.train --manifest real_data/manifest.csv --out runs/ct_hybrid --model hybrid --epochs 30 --data-kind prepared-ct
python -m src.train --manifest real_data/manifest.csv --out runs/ct_classical --model classical --epochs 30 --data-kind prepared-ct
```

No LIDC-IDRI data or clinical checkpoint is supplied. DICOM resampling, annotation matching and label adjudication remain external preparation steps, not implemented capabilities. The loader does not infer malignancy from filenames. Synthetic mode is explicitly marked; the caller must truthfully declare data provenance.

## Limitations and next semester

The historical 88.5%, 90.2% and 94.8% figures are unverified archival claims and are not current results. No quantum-only comparator, quantum advantage, noise tolerance, lower overfitting or clinical accuracy is established. This is a CPU simulator, not quantum hardware.

Next semester: patient-reviewed LIDC-IDRI evaluation, matched 8-feature classical ablation, Grad-CAM, circuit-depth/noise analysis and multiple-seed uncertainty estimates. Multimodal/federated learning and hardware deployment remain longer-term ideas.

## Source and audit

`baseline_src/` preserves the original four-file implementation from the pre-Semester-5 repository state. `docs/AUDIT.md` records the corrections and scope boundary. The current GitHub source is the Semester 5 continuation; the CI workflow verifies tests and a one-epoch synthetic smoke-training run on pushes and pull requests.
