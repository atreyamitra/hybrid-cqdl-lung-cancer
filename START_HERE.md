# START HERE

Hybrid Classical–Quantum Deep Learning for Lung Cancer Detection. Semester 5 PBL, Manipal University Jaipur, 2026–27.

Every command runs from this folder (the repository root). Python 3.12 is required. Internet is needed for the first install only.

## 1. Set up (once)

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
source .venv/bin/activate              # macOS / Linux
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If the PyTorch CPU index is blocked on your network, use `python -m pip install torch==2.14.0` instead. That is a larger download, but it also runs on a CPU and was used for the tested environment.

## 2. Test

```bash
python -m unittest discover -s tests -v
```

Expected: `Ran 39 tests` … `OK`.

## 3. Data, checkpoints and the experiment

The submission ZIP already contains `demo_data/`, all checkpoints and all results, so you can skip this step there. After a fresh `git clone`, run:

```bash
python -m src.demo_data --out demo_data
python -m src.experiments run --manifest demo_data/manifest.csv --out experiments/endsem_synthetic --data-kind synthetic --configs matched hybrid_L1 hybrid_L3 classical
```

This makes 12 runs (4 configurations × seeds 42, 43, 44). It took 17 s on a 2-core Linux CPU. The runner refuses to overwrite completed runs; add `--resume` to fill in only missing ones.

To recreate the original single-seed runs in `results/` (optional), write them to new folders:

```bash
python -m src.train --manifest demo_data/manifest.csv --out runs/hybrid --model hybrid --epochs 5 --data-kind synthetic
python -m src.train --manifest demo_data/manifest.csv --out runs/classical --model classical --epochs 5 --data-kind synthetic
```

## 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501. On Windows you can double-click `run_app.bat` instead; it sets up the environment, runs the experiment if it is missing, and starts the app. (The launcher has not been tested on Windows.)

- **Inference** page: the default checkpoint `results/hybrid/best.pt` with `fixture_23.npy` shows a score of 0.5108, threshold 0.50 and class 1. Upload a `.png` to see the rejection message.
- **Experiment results** page: the aggregate table, every run, confusion matrices and training curves.

## 5. Command-line inference and diagnostics

```bash
python -m src.predict --checkpoint results/hybrid/best.pt --image demo_data/fixture_23.npy
python -m src.predict --checkpoint experiments/endsem_synthetic/hybrid_L1/seed_43/best.pt --image demo_data/fixture_23.npy
python -m src.diagnose --checkpoint results/hybrid/best.pt --manifest demo_data/manifest.csv
```

Expected: the first prints `score_class_1` 0.5108… with `q_layers` 3 and `calibrated_probability` false. The third prints `expval_std_across_images_mean` 0.0144.

## 6. Rebuild figures and the static page (optional)

```bash
python scripts/make_figures.py
python scripts/build_demo.py            # add --embed for a single portable HTML file
```

## Remember when presenting

All results come from 24 synthetic disc images, with 4 test images per run. They show that the code works and is reproducible. They are not lung-cancer accuracy, not a model ranking and not evidence of quantum advantage. Scores are uncalibrated sigmoid outputs.
