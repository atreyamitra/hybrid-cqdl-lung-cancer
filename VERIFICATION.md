# Verification summary — end-semester, November 2026

Environment for every executed check: Linux, CPU only (2 cores), Python 3.12.3, torch 2.14.0 (PyPI build, run on CPU), PennyLane 0.45.1, NumPy 2.5.3, matplotlib 3.10.8, scikit-learn 1.8.0, Streamlit 1.67.0. Nothing below was run on Windows, on GitHub Actions or on real CT data.

## End-semester additions: executed (commands run, output checked)

| Check | Result | Evidence |
|---|---|---|
| Full test suite (`python -m unittest discover -s tests -v`) | **39/39 pass**, no warnings other than Streamlit's bare-mode notice | `test_results.txt` |
| 1- and 3-layer hybrid construction, finite non-zero gradients, invalid depths rejected | pass | `tests/test_endsem.py` |
| Checkpoint depth round-trip (L=1, L=3), legacy checkpoint without depth loads as L=3, mislabelled depth gives a clear error | pass | `tests/test_endsem.py` |
| Matched comparator: output and bottleneck shapes, value ranges, 479,705 parameters, 72 in the middle block, same extractor shapes as the hybrid | pass | `tests/test_endsem.py` |
| Experiment `experiments/endsem_synthetic/`: 4 configurations × seeds 42/43/44, 5 epochs, fixed 16/4/4 split | 12/12 runs completed in about 17 s in total; aggregate written | `aggregate.json`, `runs.csv`, per-run `run.json` |
| Same split for every run | all 12 `split_manifest.csv` files identical | test + `cmp` |
| Seed 42 of `hybrid_L3` and `classical` against the original single-seed runs in `results/` | identical metrics, confusion matrices and selected epochs | compared JSON |
| Aggregation with undefined metrics (precision undefined when no positives are predicted) | mean/std over defined values, `n_defined` reported, strict JSON (no NaN) | `tests/test_endsem.py`; `hybrid_L1` precision n=2/3 |
| Quantum-layer diagnostic on all six hybrid runs | expectation-value spread 0.008–0.024 (L=3) | per-run `diagnostic.json` |
| Runner refuses to overwrite completed runs; `--resume` skips them | pass | `tests/test_endsem.py` |
| `src.predict` on an L=1 experiment checkpoint and on `results/hybrid/best.pt` | outputs include `q_layers`; hybrid score for `fixture_23.npy` 0.5108 → class 1; `calibrated_probability: false` | console run |
| App inputs: PNG, DICOM, empty, non-NumPy bytes, wrong shape, 8-bit range, pickled object array, NaN | each rejected with a specific message | `tests/test_endsem.py` |
| App inference determinism (same checkpoint and crop twice) | identical scores; threshold 0.5; class matches score | `tests/test_endsem.py` |
| Both app pages rendered with Streamlit's `AppTest` | no exceptions | `tests/test_endsem.py` |
| App launched headless (`streamlit run app/streamlit_app.py`) | health endpoint returned `ok` | console run |
| Screenshots of the running app (inference, rejection message, results page) | captured with headless Chrome from the running app; used unedited (cropped only) in the deck | screenshot files |
| `scripts/make_figures.py`, `scripts/build_demo.py` | regenerate all figures and `DEMO.html` from saved evidence | console run |
| End-semester deck `PBL_Sem5_Endsem_Final.pptx` | schema validation passed; rendered to PDF; all 18 slides inspected; overlaps on slides 9–11 fixed and re-checked | rendering |
| Fresh extraction of the final ZIP, then the tests and predictions from the extracted copy | 39/39 pass; hybrid L=3 score 0.5108 → class 1, L=1 seed-43 score 0.5010; no absolute local paths in any JSON, CSV, Markdown or HTML file | console run (packaging step) |

## November refinement: executed earlier (unchanged since)

| Check | Result |
|---|---|
| Original 12 tests on the September code, before any change | 12/12 pass |
| Re-run of both September experiments with the September code | every metric identical; max loss/score difference 6×10⁻⁸ |
| `python -m src.demo_data` regenerates `demo_data/` | byte-identical to the shipped files |
| Probes of realistic bad inputs on the September code | whitespace patient ID accepted (leak); short row → AttributeError; PNG/missing file/bare state dict → tracebacks; all fixed (see `docs/AUDIT.md`) |
| Final runs of both models with the November code | metrics identical to September (`results/reproduction_check.json`) |
| `src.diagnose` on the hybrid checkpoint | expectation-value spread 0.0144; scores 0.5056–0.5154 (`results/hybrid/diagnostic.json`) |

CSV evidence files were converted from CRLF to LF line endings in this task so that they match what the writer now produces. Values are unchanged; re-aggregation produced identical tables.

## Not executed

| Item | Status |
|---|---|
| `run_app.bat`, `run_demo.bat` on Windows | Written and reviewed; never run on Windows. `.gitattributes` keeps them CRLF |
| `.github/workflows/ci.yml` on GitHub Actions | Updated locally (adds prediction, diagnostic and a one-epoch experiment) but **not pushed**: the GitHub token used for this branch lacks the `workflow` permission and GitHub returned 403. The file is in the ZIP; the branch keeps the September workflow. No CI run has been observed |
| Installing from the PyTorch CPU index (`download.pytorch.org`) | Blocked from the build environment; the PyPI build of the same version was used |
| `DEMO.html` in a desktop browser | Generated and checked as text; not opened in a desktop browser |
| Python 3.13, macOS or other OS/version combinations | Not tested |
| Real CT data, clinical metrics, calibration, quantum hardware or noise | Not available; not claimed |

## What the evidence does not show

Every result comes from 24 synthetic disc images with 4 test images per run. The experiment shows that each configuration trains, saves, reloads and evaluates reproducibly. It does not rank the models, does not show quantum advantage, and says nothing about lung-cancer detection. Scores are uncalibrated sigmoid outputs. Training times are local wall-clock observations on one machine, not benchmarks.
