"""Local app for the Hybrid CQDL PBL prototype.

    streamlit run app/streamlit_app.py

Page 1 scores a prepared 128x128 crop with a saved checkpoint.
Page 2 shows saved experiment results read from disk.
The app never trains, fine-tunes or changes a model.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src import app_core as core
from src.predict import load_model

torch.set_num_threads(2)
st.set_page_config(page_title='Hybrid CQDL · local app', layout='wide')

SYNTHETIC_NOTE = ('Synthetic geometric fixtures only (bright discs on noise). Nothing here is a CT image, '
                  'a clinical result or a diagnosis.')
SCORE_NOTE = ('The score is the sigmoid output of the selected model. It is **not** a calibrated probability '
              'of cancer: no calibration was performed, and the model has only seen synthetic shapes.')


@st.cache_data(show_spinner=False, ttl=30)
def cached_checkpoints():
    return core.discover_checkpoints()


@st.cache_resource(show_spinner='Loading checkpoint…')
def cached_model(path, mtime):
    return load_model(path)


@st.cache_data(show_spinner=False)
def cached_samples():
    return core.ensure_demo_samples()


def preview(x, caption):
    big = np.kron(np.clip(x, 0, 1), np.ones((3, 3), dtype=np.float32))  # 384x384, nearest-neighbour
    st.image(big, caption=caption, clamp=True, width=384)


def inference_page():
    st.header('Score a prepared crop')
    st.caption('Loads a saved checkpoint and scores one 128×128 crop at the fixed threshold 0.5. Inference only.')
    cks = [c for c in cached_checkpoints() if c['ok']]
    bad = [c for c in cached_checkpoints() if not c['ok']]
    if not cks:
        st.warning('No checkpoint found under results/, experiments/ or runs/. Create one first, for example:')
        st.code('python -m src.train --manifest demo_data/manifest.csv --out runs/app_hybrid '
                '--model hybrid --epochs 5 --data-kind synthetic')
        st.stop()
    default = next((i for i, c in enumerate(cks) if c['rel'] == 'results/hybrid/best.pt'), 0)
    choice = st.selectbox('Checkpoint', range(len(cks)), index=default, format_func=lambda i: core.checkpoint_label(cks[i]))
    c = cks[choice]
    try:
        model, ck = cached_model(str(c['path']), c['path'].stat().st_mtime)
    except ValueError as e:
        st.error(f'Could not load this checkpoint: {e}')
        st.stop()
    cols = st.columns(5)
    cols[0].metric('Model', {'hybrid': 'Hybrid', 'classical': 'Original classical', 'matched': 'Matched classical'}[ck['model']])
    cols[1].metric('Quantum layers', ck['q_layers'] if ck['q_layers'] is not None else '—')
    cols[2].metric('Training data', ck.get('data_kind', '?'))
    cols[3].metric('Selected epoch', ck.get('epoch', '?'))
    cols[4].metric('Seed', ck.get('seed', '?'))
    if ck.get('data_kind') == 'synthetic':
        st.warning('**Synthetic checkpoint.** ' + SYNTHETIC_NOTE)
    if bad:
        with st.expander(f'{len(bad)} checkpoint file(s) could not be read'):
            for b in bad:
                st.write(f'`{b["rel"]}`: {b["error"]}')

    st.subheader('Input')
    source = st.radio('Input source', ['Bundled synthetic sample', 'Upload a prepared .npy crop'], horizontal=True)
    x, caption, truth = None, '', None
    if source == 'Bundled synthetic sample':
        rows = cached_samples()
        names = [f'{r["path"]}  ·  label {r["label"]}  ·  {r["split"]} split' for r in rows]
        default_sample = next((i for i, r in enumerate(rows) if r['path'] == 'fixture_23.npy'), 0)
        i = st.selectbox('Synthetic fixture (demonstration input, not a CT image)', range(len(rows)), index=default_sample,
                         format_func=lambda k: names[k])
        r = rows[i]
        x = core.crop_from_upload(r['path'], Path(r['abs_path']).read_bytes())
        caption, truth = f'{r["path"]} (synthetic, label {r["label"]}, {r["split"]} split)', int(r['label'])
        if r['split'] != 'test':
            st.info(f'This fixture is in the **{r["split"]}** split, so the model may have seen it during training or epoch selection.')
    else:
        up = st.file_uploader('Prepared crop: a NumPy .npy file holding a 128×128 array with values in [0, 1]',
                              accept_multiple_files=False)
        st.caption('Photographs, PNG/JPEG images and DICOM files are rejected, not converted.')
        if up is None:
            st.stop()
        try:
            x = core.crop_from_upload(up.name, up.getvalue())
            caption = f'{up.name} (uploaded)'
        except ValueError as e:
            st.error(f'Input rejected: {e}')
            st.stop()

    left, right = st.columns([1, 1.25])
    with left:
        preview(x, caption)
    with right:
        res = core.run_inference(model, ck, x)
        st.subheader('Model output')
        m = st.columns(3)
        m[0].metric('Model score (sigmoid)', f'{res["score"]:.4f}')
        m[1].metric('Threshold (fixed)', f'{res["threshold"]:.2f}')
        m[2].metric('Predicted class', res['predicted_class'])
        st.write(f'**Architecture:** {res["description"]}')
        if truth is not None:
            ok = truth == res['predicted_class']
            (st.success if ok else st.error)(f'Fixture label {truth}: prediction is {"correct" if ok else "incorrect"} for this synthetic sample.')
        st.info(SCORE_NOTE)
        st.caption('Inference runs in eval mode under torch.inference_mode() with gradients disabled, '
                   'so the same checkpoint and input always give the same score.')


def table_cell(s, digits=3):
    if s['mean'] is None:
        return 'undefined'
    txt = f'{s["mean"]:.{digits}f}'
    if s['std'] is not None:
        txt += f' ± {s["std"]:.{digits}f}'
    if s['n_defined'] < s['n_runs']:
        txt += f' (n={s["n_defined"]}/{s["n_runs"]})'
    return txt


def results_page():
    st.header('Experiment results')
    st.caption('Read from saved result files on disk. Nothing on this page is typed in by hand.')
    exps = core.list_experiments()
    if exps:
        e = st.selectbox('Experiment', exps, format_func=lambda p: p.relative_to(core.ROOT).as_posix())
        data = core.load_experiment(e)
        spec, agg, runs = data['spec'], data['aggregate'], data['runs']
        st.warning('**' + ', '.join(sorted({r['data_kind'] for r in runs})).capitalize() + ' data.** ' + agg['note'])
        splits = runs[0]['splits'] or {}
        c = st.columns(6)
        c[0].metric('Seeds', ', '.join(map(str, spec.get('seeds', []))))
        c[1].metric('Epochs (max)', spec.get('epochs', '?'))
        c[2].metric('Learning rate', spec.get('lr', '?'))
        c[3].metric('Batch', spec.get('batch_size', '?'))
        c[4].metric('Split (images)', ' / '.join(str(splits.get(k, {}).get('images', '?')) for k in ('train', 'val', 'test')))
        c[5].metric('Threshold', spec.get('threshold', 0.5))
        st.caption('The same split is used for every run; seeds change only initialisation and batch order. '
                   'Checkpoints are selected on validation loss.')

        st.subheader('Aggregate across seeds (mean ± sample std)')
        st.dataframe(pd.DataFrame([{
            'Configuration': a['label'], 'Parameters': f'{a["parameters"]:,}', 'Runs': a['n_runs'],
            'Accuracy': table_cell(a['accuracy']), 'Precision': table_cell(a['precision']), 'Recall': table_cell(a['recall']),
            'Specificity': table_cell(a['specificity']), 'F1': table_cell(a['f1']), 'ROC-AUC': table_cell(a['roc_auc']),
            'Train s': table_cell(a['train_seconds'], 2),
            'Predicted positive (of 4)': ', '.join(map(str, a['predicted_positive_per_run'])),
        } for a in agg['configs']]), hide_index=True, width='stretch')

        st.subheader('Individual runs')
        st.dataframe(pd.DataFrame([{
            'Configuration': r['label'], 'Seed': r['seed'], 'Parameters': f'{r["parameters"]:,}',
            'Selected epoch': r['selected_epoch'], **{k.replace('_', '-').upper() if k == 'roc_auc' else k.capitalize(): core.fmt(r[k]) for k in
                                                       ('accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc')},
            'Confusion [[TN,FP],[FN,TP]]': str(r['confusion_matrix']), 'Train time (s)': core.fmt(r['train_seconds'], 2),
        } for r in runs]), hide_index=True, width='stretch')

        st.subheader('Inspect one run')
        k = st.selectbox('Run', range(len(runs)), format_func=lambda i: f'{runs[i]["label"]} · seed {runs[i]["seed"]}')
        r = runs[k]
        cols = st.columns(2)
        if 'training_curves' in r['plots']:
            cols[0].image(str(r['plots']['training_curves']), caption='Training and validation loss')
        if 'confusion_matrix' in r['plots']:
            cols[1].image(str(r['plots']['confusion_matrix']), caption='Test confusion matrix at threshold 0.5', width=360)
        if r['diagnostic']:
            dg = r['diagnostic']
            st.write(f'**Quantum-layer diagnostic** (all 24 fixtures, `src.diagnose`): expectation-value spread '
                     f'{dg["expval_std_across_images_mean"]}, classifier |w| {dg["classifier_weight_abs_mean"]}, '
                     f'scores {dg["score_min"]}–{dg["score_max"]}.')
        if r['runtime']:
            rt = r['runtime']
            st.caption(f'Measured on: {rt.get("platform")} · {rt.get("cpu_count")} CPU cores · {rt.get("torch_threads")} torch threads · CPU only. '
                       'Training time is a local observation, not a benchmark.')
    else:
        st.info('No multi-seed experiment found. Run: python -m src.experiments run --manifest demo_data/manifest.csv '
                '--out experiments/endsem_synthetic --data-kind synthetic')

    single = core.load_single_results()
    if single:
        st.subheader('Original single-seed runs (results/, seed 42)')
        st.dataframe(pd.DataFrame([{
            'Run': s['name'], 'Architecture': s['metrics'].get('architecture', s['metrics']['model']),
            'Parameters': f'{s["metrics"]["parameters"]:,}', 'Data': s['metrics']['data_kind'],
            'Split (images)': ' / '.join(str(s['run'].get('splits', {}).get(k, {}).get('images', '?')) for k in ('train', 'val', 'test')),
            **{k: core.fmt(s['metrics'][k]) for k in ('accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc')},
            'Confusion': str(s['metrics']['confusion_matrix']),
        } for s in single]), hide_index=True, width='stretch')


st.sidebar.title('Hybrid CQDL')
st.sidebar.caption('Semester 5 PBL · Manipal University Jaipur · 2026–27')
page = st.sidebar.radio('Page', ['Inference', 'Experiment results'])
st.sidebar.warning('Research prototype. Synthetic evidence only. Not a diagnostic tool.')
inference_page() if page == 'Inference' else results_page()
