# Report update note

The 26-page report (`PBL_Sem5_Report.docx`, November 2026) is correct about everything it covers, but it predates the end-semester additions. If the report is resubmitted, change these sections. Every number below comes from `experiments/endsem_synthetic/`.

| Report section | What to change |
|---|---|
| Abstract | Add one sentence each on the local app, the matched-bottleneck comparator and the 3-seed experiment (matched 0.750 ± 0.250, hybrid L=1 0.667 ± 0.289, hybrid L=3 0.500 ± 0.000 accuracy; AUC 1.0 everywhere; synthetic, 4 test images per run). The test count is now 39. |
| 4.1 / 4.3 | Add the matched comparator: Linear(8→8)+tanh replaces the quantum layer; 479,705 parameters. Add configurable depth: 1-layer hybrid has 479,641 parameters. |
| 5.2 Code structure | Add `src/models.py`, `src/matched_classical.py`, `src/experiments.py`, `src/app_core.py`, `app/streamlit_app.py`, `run_app.bat`. |
| 5.5 Demonstration | The Streamlit app is now the main demonstration; `DEMO.html` is a static summary. |
| 6.1 Tests | 39 tests; add the rows from `tests/test_endsem.py`. |
| Chapter 7 | Add a section on the 3-seed experiment: the table from README "Results", the per-seed curve figure (`docs/figures/fig_seed_val_loss.png`) and the diagnostic spreads (L=3: 0.008–0.024; L=1: 0.022–0.082). |
| Chapter 9 Roadmap | Items 2 (matched comparator) and 3 (multiple seeds) are now done on synthetic data. The remaining work is running them on real data with more test cases. |
| Chapter 3 | Add a fourth stage: end-semester finalization (branch `endsem-finalization`). |

Nothing in the report needs to be withdrawn.
