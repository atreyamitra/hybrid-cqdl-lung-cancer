"""Build the offline DEMO.html evidence page from saved results.

Run from the project root:
    python scripts/build_demo.py           # small page that links to the PNG files (default)
    python scripts/build_demo.py --embed   # single portable file with images embedded
The page only displays saved results; the interactive app is app/streamlit_app.py.
"""
import base64, csv, html, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'


EMBED = '--embed' in sys.argv[1:]


def img(path, alt):
    path = Path(path)
    if not EMBED:
        return f'<img alt="{html.escape(alt)}" src="{path.relative_to(ROOT).as_posix()}">'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def j(path): return json.loads(Path(path).read_text())


m = {k: j(R / k / 'metrics.json') for k in ('classical', 'hybrid')}
pred = j(R / 'hybrid' / 'inference_example.json')
diag = j(R / 'hybrid' / 'diagnostic.json')
rep = j(R / 'reproduction_check.json')
tests = (ROOT / 'test_results.txt').read_text().strip().splitlines()[-3:]
ntests = next((l.split()[1] for l in tests if l.startswith('Ran ')), '?')


def f(x): return '—' if x is None else f'{x:.3f}'


rows = ''.join(
    f'<tr><td>{name}</td><td>{mm["parameters"]:,}</td><td>{mm["selected_epoch"]}</td><td>{f(mm["accuracy"])}</td>'
    f'<td>{f(mm["precision"])}</td><td>{f(mm["recall"])}</td><td>{f(mm["specificity"])}</td><td>{f(mm["f1"])}</td>'
    f'<td>{f(mm["roc_auc"])}</td><td>{mm["confusion_matrix"]}</td></tr>'
    for name, mm in [('Classical CNN', m['classical']), ('Hybrid CNN + quantum layer', m['hybrid'])])
preds = ''
for k in ('classical', 'hybrid'):
    for r in csv.DictReader((R / k / 'predictions.csv').open()):
        preds += f'<tr><td>{k}</td><td>{html.escape(r["path"])}</td><td>{r["label"]}</td><td>{float(r["probability"]):.4f}</td></tr>'

F = ROOT / 'docs' / 'figures'
EXP = ROOT / 'experiments' / 'endsem_synthetic' / 'aggregate.json'
exp_section = ''
if EXP.is_file():
    def cell(x):
        if x['mean'] is None: return 'undefined'
        t = f"{x['mean']:.3f}" + (f" ± {x['std']:.3f}" if x['std'] is not None else '')
        return t + (f" (n={x['n_defined']}/{x['n_runs']})" if x['n_defined'] < x['n_runs'] else '')
    agg = json.loads(EXP.read_text())
    rows_e = ''.join(f"<tr><td>{html.escape(c['label'])}</td><td>{c['parameters']:,}</td><td>{c['n_runs']}</td>" +
                     ''.join(f"<td>{cell(c[k])}</td>" for k in ('accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc')) +
                     f"<td>{cell(c['train_seconds'])}</td></tr>" for c in agg['configs'])
    exp_section = (f'<h2 id="exp">End-semester experiment: 3 seeds × 4 configurations</h2><p class="note">{html.escape(agg["note"])}</p>'
                   '<table><tr><th>Configuration</th><th>Parameters</th><th>Runs</th><th>Accuracy</th><th>Precision</th><th>Recall</th>'
                   f'<th>Specificity</th><th>F1</th><th>AUC</th><th>Train s</th></tr>{rows_e}</table>'
                   '<p>Mean ± sample standard deviation across seeds 42, 43, 44 on one fixed 16/4/4 split. Source: <code>experiments/endsem_synthetic/</code>.</p>')
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid CQDL: saved evidence (Semester 5)</title>
<style>
body{{font:17px/1.6 system-ui,Segoe UI,Arial,sans-serif;color:#172a3b;background:#f6f8fa;max-width:1080px;margin:32px auto;padding:0 24px}}
h1{{font-size:34px;line-height:1.2;margin-bottom:4px}} h2{{margin-top:44px;border-top:1px solid #cbd5df;padding-top:24px}}
table{{width:100%;border-collapse:collapse;background:#fff;font-size:15px}} td,th{{padding:8px 10px;text-align:left;border:1px solid #cbd5df}}
th{{background:#eef3f8}} img{{max-width:100%;background:#fff;margin:8px 0;border:1px solid #e1e7ee}}
.note{{background:#fff7e6;border:1px solid #f0d49a;padding:12px 16px;border-radius:6px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} pre{{background:#fff;padding:14px;border:1px solid #e1e7ee;overflow:auto;font-size:14px}}
nav a{{margin-right:14px;color:#165d98}} small{{color:#52606d}}
</style></head><body>
<h1>Hybrid Classical–Quantum Deep Learning for Lung Cancer Detection</h1>
<p><small>Semester 5 PBL · Manipal University Jaipur · Atreya Mitra (2427030002) and Debraj Mondal (2427030392) · Section H · Guide: Mr Tapan Kumar Dey · 2026–27</small></p>
<p class="note"><strong>Offline evidence viewer.</strong> This page displays saved results from completed local runs on <strong>24 synthetic geometric fixtures</strong>. It performs no inference in the browser, uses no patient data and makes no diagnostic claim. Run the Python CLI for live local inference.</p>
<p><strong>Interactive version:</strong> <code>streamlit run app/streamlit_app.py</code> scores crops with saved checkpoints and shows every experiment result. This static page needs no installation.</p>
<nav><a href="#arch">Architecture</a><a href="#exp">3-seed experiment</a><a href="#results">Results</a><a href="#why">Why the hybrid scores sit near 0.5</a><a href="#repro">Reproducibility</a><a href="#next">Next semester</a></nav>

<h2 id="arch">What is implemented</h2>
<p>A PyTorch CNN feature extractor feeds an eight-feature projection into an eight-qubit variational quantum layer (PennyLane, analytic <code>default.qubit</code> simulator, backpropagation). The classical comparator keeps the original 128-unit dense head. This is a CNN with a variational quantum layer, not a full quantum convolution/pooling network.</p>
{img(F / 'fig_architecture.png', 'Hybrid architecture block diagram')}
{img(F / 'fig_circuit_gates.png', 'Eight-qubit circuit expanded to RX and CNOT gates')}
{img(F / 'fig_fixtures.png', 'Example synthetic fixtures')}

<h2 id="results">Held-out results on 4 synthetic test fixtures</h2>
<p>Seed 42, Adam learning rate 0.0001, batch 4, at most 5 epochs, checkpoint chosen by lowest validation loss, fixed threshold 0.5. Four test examples cannot measure lung-cancer detection; these numbers verify that the pipeline runs end to end.</p>
<table><tr><th>Model</th><th>Parameters</th><th>Epoch</th><th>Acc.</th><th>Prec.</th><th>Recall</th><th>Spec.</th><th>F1</th><th>AUC</th><th>Confusion [[TN,FP],[FN,TP]]</th></tr>{rows}</table>
<p>The parameter difference comes from the 128-unit versus 8-unit dense bottleneck, not from a demonstrated quantum efficiency gain.</p>
{img(F / 'fig_loss_shared_axis.png', 'Loss curves on a shared axis')}
{img(F / 'fig_test_scores.png', 'Test scores against the 0.5 threshold')}
<table><tr><th>Model</th><th>Test crop</th><th>Label</th><th>Score</th></tr>{preds}</table>
<div class="grid">{img(R / 'classical' / 'confusion_matrix.png', 'Classical confusion matrix')}{img(R / 'hybrid' / 'confusion_matrix.png', 'Hybrid confusion matrix')}</div>

<h2 id="why">Why the hybrid scores sit near 0.5</h2>
<p>The diagnostic CLI reads the trained hybrid checkpoint and measures the signal at the quantum layer for all 24 fixtures. Encoded angles reach about 1 radian, but the eight Pauli-Z expectation values vary across images by a standard deviation of only {diag["expval_std_across_images_mean"]} on a −1 to +1 scale, and the Linear(8,1) weights average {diag["classifier_weight_abs_mean"]} in magnitude. The classifier therefore receives nearly the same input for every image, so every score lies between {diag["score_min"]} and {diag["score_max"]}. This is an observation about this short synthetic run, not a general property of quantum layers.</p>
{img(F / 'fig_hybrid_expvals.png', 'Spread of expectation values across fixtures')}
<pre>python -m src.diagnose --checkpoint results/hybrid/best.pt --manifest demo_data/manifest.csv</pre>

<h2>Reloaded-checkpoint prediction</h2>
<pre>python -m src.predict --checkpoint results/hybrid/best.pt --image demo_data/fixture_23.npy
{html.escape(json.dumps(pred, indent=2))}</pre>
<p>The score is a sigmoid output of a model trained on synthetic shapes. It is not a calibrated probability and not a medical risk.</p>

<h2 id="repro">Reproducibility check</h2>
<p>The September 2026 runs were repeated with the November 2026 code. Every metric, confusion matrix and selected epoch is identical; losses and scores differ by at most {rep["models"]["hybrid"]["max_abs_test_score_difference"]:.1e} (float32 rounding). Elapsed time changed from {rep["models"]["classical"]["elapsed_seconds_sept"]} s / {rep["models"]["hybrid"]["elapsed_seconds_sept"]} s to {rep["models"]["classical"]["elapsed_seconds_nov"]} s / {rep["models"]["hybrid"]["elapsed_seconds_nov"]} s (classical / hybrid), so runtime is not a benchmark. Automated tests: {ntests} passed.</p>

{exp_section}
<h2 id="next">Reserved for next semester</h2>
<p>Verified real CT cohort and preprocessing pipeline, real-data training and evaluation of all four configurations, Grad-CAM, quantum noise studies, hyperparameter search, uncertainty analysis with more test data, and external validation. Quantum hardware, multimodal and federated learning remain longer-term directions. The historical 88.5%, 90.2% and 94.8% figures were never reproduced and are not results of this project.</p>
</body></html>'''
(ROOT / 'DEMO.html').write_text(page, encoding='utf-8')
print(ROOT / 'DEMO.html', len(page))
