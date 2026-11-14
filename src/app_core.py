"""Logic behind the local Streamlit app, kept free of Streamlit so it can be tested.

Inference only: checkpoints are loaded with weights-only loading, every
parameter has requires_grad disabled, and scoring runs in eval mode under
torch.inference_mode(). Nothing in the app trains or modifies a model.
"""
import csv, io, json
from pathlib import Path
import numpy as np
from .data import validate_crop
from .demo_data import create_demo
from .predict import load_model, score_array, THRESHOLD
from .models import describe

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRS = ('results', 'experiments', 'runs')
MAX_UPLOAD_BYTES = 2_000_000  # a valid 128x128 float64 .npy is ~131 kB


def read_metadata(path):
    """Checkpoint metadata without building the model (weights-only load)."""
    import torch
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as e:
        raise ValueError(f'{path}: not a readable weights-only checkpoint ({e.__class__.__name__})') from None
    if not isinstance(ck, dict) or not {'state_dict', 'model', 'data_kind'} <= set(ck):
        raise ValueError(f'{path}: not a checkpoint produced by src.train')
    q = ck.get('q_layers', 3 if ck['model'] == 'hybrid' else None)
    return dict(model=ck['model'], q_layers=q, data_kind=ck['data_kind'], epoch=ck.get('epoch'), seed=ck.get('seed'))


def discover_checkpoints(root=ROOT):
    """Find saved checkpoints; returns metadata dicts, unreadable files flagged."""
    found = []
    for base in CHECKPOINT_DIRS:
        for path in sorted((Path(root) / base).glob('**/best.pt')):
            if 'archive' in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            try:
                ck = read_metadata(path)
            except ValueError as e:
                found.append(dict(path=path, rel=rel, ok=False, error=str(e)))
                continue
            found.append(dict(path=path, rel=rel, ok=True, model=ck['model'], q_layers=ck['q_layers'],
                              data_kind=ck.get('data_kind'), epoch=ck.get('epoch'), seed=ck.get('seed'),
                              description=describe(ck['model'], ck['q_layers'])))
    return found


def checkpoint_label(c):
    if not c['ok']:
        return f'{c["rel"]}  (unreadable)'
    return f'{c["rel"]}  ·  {c["description"]}  ·  seed {c["seed"]}  ·  {c["data_kind"]}'


def ensure_demo_samples(root=ROOT):
    """Return the synthetic fixture rows, generating them deterministically if
    absent (src.demo_data reproduces the shipped files byte for byte)."""
    manifest = Path(root) / 'demo_data' / 'manifest.csv'
    if not manifest.is_file():
        create_demo(manifest.parent)
    rows = list(csv.DictReader(manifest.open()))
    for r in rows:
        r['abs_path'] = manifest.parent / r['path']
    return rows


def crop_from_upload(name, data):
    """Validate an uploaded file and return a float32 128x128 crop.

    Only prepared .npy crops are accepted. Photographs, DICOM files and other
    images are rejected rather than converted, because converting them would
    pretend to support an input the model was never designed for."""
    if not name or not name.lower().endswith('.npy'):
        suffix = Path(name or '').suffix or 'no extension'
        raise ValueError(f'"{name}" is not a prepared .npy crop ({suffix}). This app does not convert photographs, '
                         'PNG/JPEG images or DICOM scans; prepare a 128x128 crop scaled to [0, 1] and save it with numpy.save.')
    if len(data) == 0:
        raise ValueError(f'"{name}" is empty')
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f'"{name}" is {len(data):,} bytes; a 128x128 crop should be about 65-131 kB')
    if not data.startswith(b'\x93NUMPY'):
        raise ValueError(f'"{name}" is not a readable .npy array (missing the NumPy file header)')
    try:
        x = np.load(io.BytesIO(data), allow_pickle=False)
    except ValueError as e:
        if 'pickle' in str(e).lower():
            raise ValueError(f'"{name}" contains pickled Python objects; only plain numeric arrays are accepted') from None
        raise ValueError(f'"{name}" is not a readable .npy array') from None
    except Exception:
        raise ValueError(f'"{name}" is not a readable .npy array') from None
    return validate_crop(x, f'"{name}"')


def run_inference(model, ck, x):
    score = score_array(model, x)
    return dict(score=score, threshold=THRESHOLD, predicted_class=int(score >= THRESHOLD),
                model=ck['model'], q_layers=ck['q_layers'], description=describe(ck['model'], ck['q_layers']),
                data_kind=ck.get('data_kind'), epoch=ck.get('epoch'), seed=ck.get('seed'))


# ---------- experiment results ----------

def list_experiments(root=ROOT):
    return [p for p in sorted((Path(root) / 'experiments').glob('*')) if (p / 'aggregate.json').is_file()]


def load_experiment(path):
    from .experiments import collect_runs
    path = Path(path)
    agg = json.loads((path / 'aggregate.json').read_text())
    spec = json.loads((path / 'experiment.json').read_text()) if (path / 'experiment.json').is_file() else {}
    runs = collect_runs(path)
    for r in runs:
        d = path / r['run_dir']
        r['plots'] = {k: d / f'{k}.png' for k in ('training_curves', 'confusion_matrix', 'roc_curve') if (d / f'{k}.png').is_file()}
        rj = d / 'run.json'
        r['runtime'] = json.loads(rj.read_text()).get('runtime') if rj.is_file() else None
        dj = d / 'diagnostic.json'
        r['diagnostic'] = json.loads(dj.read_text()) if dj.is_file() else None
    return dict(path=path, aggregate=agg, spec=spec, runs=runs)


def load_single_results(root=ROOT):
    """Original single-seed evidence in results/<model>/ (September protocol)."""
    out = []
    for d in sorted((Path(root) / 'results').glob('*')):
        if (d / 'metrics.json').is_file() and d.name != 'archive_2026-09':
            m = json.loads((d / 'metrics.json').read_text())
            r = json.loads((d / 'run.json').read_text()) if (d / 'run.json').is_file() else {}
            out.append(dict(name=d.name, metrics=m, run=r,
                            plots={k: d / f'{k}.png' for k in ('training_curves', 'confusion_matrix', 'roc_curve') if (d / f'{k}.png').is_file()}))
    return out


def fmt(v, digits=3):
    if v is None:
        return 'undefined'
    return f'{v:.{digits}f}' if isinstance(v, float) else str(v)
