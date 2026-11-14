"""Reproducible CPU training and held-out evaluation for prepared crops."""
import argparse, csv, hashlib, json, math, os, platform, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data import load_manifest, assign_splits, load_crop, SPLITS
from .metrics import binary_metrics
from .models import MODEL_TYPES, build_model, describe, count_parameters


class Crops(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = torch.from_numpy(load_crop(r['path'])).unsqueeze(0)
        return x, torch.tensor([r['label']], dtype=torch.float32)


def evaluate(model, loader):
    model.eval(); ys = []; ps = []; losses = []
    with torch.no_grad():
        for x, y in loader:
            p = model(x); losses.append((torch.nn.functional.binary_cross_entropy(p, y).item(), len(y)))
            ys.extend(y.flatten().tolist()); ps.extend(p.flatten().tolist())
    return sum(a * b for a, b in losses) / sum(b for _, b in losses), ys, ps


def portable(path_arg):
    """Relative paths are kept as typed; absolute ones are reduced to their
    final component so evidence never records the training machine's folders."""
    p = Path(path_arg)
    return p.name if p.is_absolute() else p.as_posix()


def evidence_path(manifest_arg, relpath):
    """Crop path written to evidence files: relative to where the command was
    run when the manifest was given relatively, else relative to the manifest."""
    m = Path(manifest_arg)
    return Path(relpath).as_posix() if m.is_absolute() else (m.parent / relpath).as_posix()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
    p.add_argument('--model', choices=list(MODEL_TYPES), default='hybrid',
                   help='hybrid (CNN + quantum layer), classical (original 128-unit head) or matched (8-feature classical bottleneck)')
    p.add_argument('--q-layers', type=int, default=None,
                   help='BasicEntanglerLayers depth for the hybrid model (default 3); not valid for other models')
    p.add_argument('--epochs', type=int, default=30); p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42, help='initialisation and batch-order seed')
    p.add_argument('--split-seed', type=int, default=42,
                   help='seed for the automatic patient split only; ignored when the manifest supplies splits')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--data-kind', choices=['synthetic', 'prepared-ct'], required=True)
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if min(a.epochs, a.batch_size, a.patience) <= 0 or a.lr <= 0: p.error('Training settings must be positive')
    if a.q_layers is not None and (a.model != 'hybrid' or a.q_layers < 1):
        p.error('--q-layers must be a positive integer and is only valid with --model hybrid')
    q_layers = (a.q_layers or 3) if a.model == 'hybrid' else None
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): p.error('Output directory must be empty to protect previous runs')
    try:
        rows = load_manifest(a.manifest)
        supplied = all(r.get('split') for r in rows)
        rows = assign_splits(rows, a.split_seed)
    except (ValueError, OSError) as e:
        p.error(f'manifest rejected: {e}')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    loaders = {s: DataLoader(Crops([r for r in rows if r['split'] == s]), batch_size=a.batch_size, shuffle=s == 'train',
                             generator=torch.Generator().manual_seed(a.seed)) for s in SPLITS}
    model = build_model(a.model, q_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    history = []; best = float('inf'); stale = 0; start = time.perf_counter()
    train_start = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train(); loss_sum = 0; n = 0
        for x, y in loaders['train']:
            optimizer.zero_grad(); prob = model(x); loss = torch.nn.functional.binary_cross_entropy(prob, y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); optimizer.step(); loss_sum += loss.item() * len(y); n += len(y)
        val_loss, _, _ = evaluate(model, loaders['val'])
        if not math.isfinite(val_loss): raise RuntimeError(f'Nonfinite validation loss at epoch {epoch}')
        history.append(dict(epoch=epoch, train_loss=loss_sum / n, val_loss=val_loss))
        print(history[-1], flush=True)
        if val_loss < best:
            best = val_loss; stale = 0
            torch.save({'state_dict': model.state_dict(), 'model': a.model, 'q_layers': q_layers, 'epoch': epoch,
                        'seed': a.seed, 'data_kind': a.data_kind}, out / 'best.pt')
        else: stale += 1
        if stale >= a.patience: break
    train_seconds = time.perf_counter() - train_start
    checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    _, y, prob = evaluate(model, loaders['test'])
    result = binary_metrics(y, prob)
    result.update(data_kind=a.data_kind, model=a.model, q_layers=q_layers, architecture=describe(a.model, q_layers),
                  seed=a.seed, selected_epoch=checkpoint['epoch'], epochs_run=len(history),
                  parameters=count_parameters(model), train_seconds=train_seconds,
                  elapsed_seconds=time.perf_counter() - start, clinical_validation=False)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    import pennylane
    metadata = dict(config=dict(vars(a), manifest=portable(a.manifest), out=portable(a.out)), split_source='manifest' if supplied else f'automatic (split_seed={a.split_seed})',
                    python=platform.python_version(), torch=torch.__version__,
                    runtime=dict(platform=platform.platform(), machine=platform.machine(),
                                 processor=platform.processor() or 'unreported', cpu_count=os.cpu_count(),
                                 torch_threads=torch.get_num_threads(), device='cpu'),
                    pennylane=pennylane.__version__, numpy=np.__version__,
                    manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                    splits={s: dict(images=sum(r['split'] == s for r in rows),
                                    patients=len({r['patient_id'] for r in rows if r['split'] == s})) for s in SPLITS})
    (out / 'run.json').write_text(json.dumps(metadata, indent=2))
    with (out / 'split_manifest.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['path', 'patient_id', 'label', 'split', 'sha256'])
        for r in rows: w.writerow([evidence_path(a.manifest, r['relpath']), r['patient_id'], r['label'], r['split'], r['sha256']])
    with (out / 'predictions.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['patient_id', 'path', 'label', 'probability'])
        for r, pr in zip([r for r in rows if r['split'] == 'test'], prob):
            w.writerow([r['patient_id'], evidence_path(a.manifest, r['relpath']), r['label'], pr])
    from .plots import save_plots
    save_plots(out, history, y, prob, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
