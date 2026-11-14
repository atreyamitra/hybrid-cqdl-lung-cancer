# Hybrid Classical–Quantum Deep Learning for Lung Cancer Detection

Semester 5 PBL, Manipal University Jaipur, 2026–27. Atreya Mitra (2427030002) and Debraj Mondal (2427030392), Section H. Guide: Mr Tapan Kumar Dey.

This is a PyTorch + PennyLane research prototype. A small CNN reduces a prepared 128×128 crop to eight numbers, an eight-qubit variational circuit (simulated) transforms them, and Linear(8,1) + sigmoid produces a score. Two classical comparators are included: the original CNN with a 128-unit head, and a **matched-bottleneck** CNN that replaces only the quantum layer with a small classical block.

The prototype classifies **prepared crops**. It does not read DICOM, locate nodules or diagnose anything. All training so far uses **synthetic geometric fixtures**.

## Quick start

See `START_HERE.md` for every command with its expected output. From the project root, using Python 3.12:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
#   if that index is blocked: python -m pip install torch==2.14.0
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.demo_data --out demo_data            # 24 synthetic fixtures (only needed after a git clone)
python -m src.experiments run --manifest demo_data/manifest.csv --out experiments/endsem_synthetic \
    --data-kind synthetic --configs matched hybrid_L1 hybrid_L3 classical
streamlit run app/streamlit_app.py                 # local app: inference + experiment results
```

On Windows, `run_app.bat` performs the same steps and opens the app. It has not been executed on Windows by the authors.

Git stores text only. Checkpoints (`*.pt`), plots (`*.png`) and fixture arrays are excluded by `.gitignore`; the submission ZIP includes them, and after a clone the commands above regenerate them in about a minute on a CPU.

## Models

| Stage | Hybrid (`--model hybrid`) | Matched classical (`--model matched`) | Original classical (`--model classical`) |
|---|---|---|---|
| Input | 1×128×128 crop | same | same |
| Conv block 1 | Conv2d(1,32,3), ReLU, MaxPool2 → 32×63×63 | same | same |
| Conv block 2 | Conv2d(32,64,3), ReLU, MaxPool2 → 64×30×30 | same | same |
| Bottleneck | Flatten 57,600 → Linear → **8** (no activation) | same | Linear → **128** + ReLU |
| Middle block | 8-qubit RX AngleEmbedding, *L* BasicEntanglerLayers (RX + CNOT ring), 8 Pauli-Z ⟨Z⟩ ∈ [−1,1] | Linear(8→8) + tanh ∈ [−1,1] | — |
| Output | Linear(8→1) + sigmoid | same | Linear(128→1) + sigmoid |
| Trainable middle-block parameters | 8·*L* angles (24 at *L* = 3) | 72 | — |
| **Total parameters** | **479,641** (*L* = 1) · **479,657** (*L* = 3) | **479,705** | **7,391,873** |

- The quantum layer runs on PennyLane's analytic `default.qubit` simulator with `diff_method="backprop"`, wrapped as a `TorchLayer`. It is a CNN with one variational layer, not a quantum convolutional network.
- Depth is set by `--q-layers` (default 3, the original architecture). Depth is saved in every checkpoint. Checkpoints written before depth was configurable have no depth field and load as three layers.
- The matched comparator differs from the hybrid **only** in the middle block. It is matched in bottleneck width and interface, not exactly in parameter count (72 vs 24 middle-block parameters).

## Commands

| Command | Purpose |
|---|---|
| `python -m src.train` | Validate a manifest, train one model, select the epoch by validation loss, evaluate on the test split at 0.5 and save the evidence |
| `python -m src.experiments run` / `aggregate` | Train several configurations × seeds on one fixed split; write per-run evidence and mean ± std tables |
| `python -m src.predict` | Score one prepared `.npy` crop with a checkpoint |
| `python -m src.diagnose` | Measure the spread of a hybrid checkpoint's eight quantum outputs across a manifest |
| `streamlit run app/streamlit_app.py` | Local app with an inference page and an experiment-results page |
| `python scripts/make_figures.py`, `python scripts/build_demo.py` | Regenerate the figures and the static `DEMO.html` from saved results |

Training options: `--model hybrid|classical|matched`, `--q-layers N` (hybrid only, default 3), `--epochs`, `--batch-size` (4), `--lr` (1e-4), `--patience` (5), `--seed` (initialisation and batch order), `--split-seed` (automatic split only) and `--data-kind synthetic|prepared-ct` (required).

## The local app

- **Inference page.** Pick any checkpoint found under `results/`, `experiments/` or `runs/`. The page shows its model, depth, training data kind, epoch and seed. You can score a bundled synthetic fixture (regenerated automatically if missing) or upload a prepared `.npy` crop. The page shows a preview, the sigmoid **model score**, the fixed threshold 0.5 and the predicted class. Synthetic checkpoints and inputs are labelled, and the score is explicitly described as not a calibrated probability. PNG/JPEG/DICOM files, pickled arrays, wrong shapes, NaNs and 8-bit ranges are rejected with a message that explains why. Inference runs in eval mode under `torch.inference_mode()` with gradients disabled; the app never trains.
- **Results page.** Reads `experiments/*/aggregate.json` and the per-run files. It shows configuration, seed, data type and split sizes, all six metrics (with "undefined" where a denominator is zero), confusion matrices, training curves, parameter counts, measured training time with the machine description, the quantum-layer diagnostic and the original single-seed runs from `results/`.

## Data contract

The manifest is a CSV file with the columns `path,patient_id,label` and an optional `split` column. Each crop is a finite 128×128 `.npy` array with values in [0,1]. The loader trims stray spaces, rejects case-variant patient IDs, duplicate paths and byte-identical crops, rejects any patient appearing in more than one split and any split missing a class, and reports the CSV line number for every error. See `docs/DATASET_PROTOCOL.md`.

## Results (synthetic fixtures only)

**End-semester experiment** (`experiments/endsem_synthetic/`): four configurations × seeds 42, 43, 44; one fixed 16/4/4 split; 5 epochs; Adam 1e-4; batch size 4; checkpoint chosen by validation loss; threshold 0.5. Values are mean ± sample std over three seeds.

| Configuration | Params | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Train s |
|---|---:|---|---|---|---|---|---|---|
| Matched classical | 479,705 | 0.750 ± 0.250 | 0.722 ± 0.255 | 1.000 ± 0.000 | 0.500 ± 0.500 | 0.822 ± 0.168 | 1.000 ± 0.000 | 0.50 ± 0.10 |
| Hybrid, 1 layer | 479,641 | 0.667 ± 0.289 | 0.750 ± 0.354 (n=2/3) | 0.667 ± 0.577 | 0.667 ± 0.577 | 0.556 ± 0.509 | 1.000 ± 0.000 | 0.69 ± 0.01 |
| Hybrid, 3 layers | 479,657 | 0.500 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.667 ± 0.000 | 1.000 ± 0.000 | 0.89 ± 0.04 |
| Original classical | 7,391,873 | 0.750 ± 0.000 | 0.778 ± 0.192 | 0.833 ± 0.289 | 0.667 ± 0.289 | 0.756 ± 0.077 | 1.000 ± 0.000 | 1.09 ± 0.06 |

- Precision is undefined for the 1-layer hybrid with seed 42 (it predicted no positives), so its mean uses 2 of 3 runs.
- Training time is wall-clock time for the training loop on a 2-core Linux CPU with 2 torch threads (recorded in each `run.json`). It is a local observation, not a benchmark.
- Seed 42 of `hybrid_L3` and `classical` reproduces the original single-seed runs in `results/` exactly.

**How to read this.** Each run has four test images, so one image moves accuracy by 0.25. ROC-AUC is 1.0 in every run because disc size makes the fixtures easy to *rank*; the thresholded metrics vary with the seed because the scores sit close to 0.5. These numbers show that every configuration trains and evaluates reproducibly. They do **not** rank the models, show quantum advantage or say anything about lung cancer.

The 3-layer hybrid labelled all four test images positive under every seed. `src.diagnose` attributes this to a narrow spread of quantum outputs across images (0.008–0.024 for the three seeds; see each run's `diagnostic.json`). The historical 88.5%, 90.2% and 94.8% figures in older material were never reproduced and are not results of this project.

## Verification

`python -m unittest discover -s tests -v` runs 39 tests (see `test_results.txt` and `VERIFICATION.md`). They cover data validation and leakage, metrics, gradients for 1- and 3-layer hybrids, checkpoint/depth round-trips, the matched comparator, experiment aggregation with undefined metrics, an end-to-end experiment run, invalid app inputs and a render of both app pages. Tested on Linux CPU: Python 3.12.3, torch 2.14.0, PennyLane 0.45.1, NumPy 2.5.3, scikit-learn 1.8.0, matplotlib 3.10.8, Streamlit 1.67.0. The Windows launchers and the GitHub Actions workflow are included but have not been executed.

## Not implemented (next semester)

Raw DICOM/CT preprocessing, LIDC-IDRI annotation matching and nodule extraction, verified clinical labels, real-cohort training and evaluation, Grad-CAM, quantum noise studies, hyperparameter search, external validation and uncertainty analysis with adequate test data. Quantum hardware, federated and multimodal learning are longer-term directions. See `CONTINUATION_HANDOFF.md`.

## History

`baseline_src/` holds the February 2026 files. The September 2026 upgrade is commit e345a73. `CHANGELOG.md` separates inherited work from the November refinement and the end-semester additions, and `results/archive_2026-09/` keeps the September evidence.
