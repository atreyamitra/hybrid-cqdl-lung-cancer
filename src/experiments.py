"""Small, reproducible multi-seed experiment runner and aggregator.

    python -m src.experiments run --manifest demo_data/manifest.csv \\
        --out experiments/endsem_synthetic --data-kind synthetic
    python -m src.experiments aggregate --out experiments/endsem_synthetic

Every configuration is trained with every seed on the SAME data split (the
split comes from the manifest, or from the fixed --split-seed), so differences
between seeds reflect training randomness only. Each run uses src.train
unchanged: validation loss selects the checkpoint and the test threshold is
fixed at 0.5. Nothing here looks at test results to make a decision.
"""
import argparse, csv, json, math, platform, statistics, sys, time
from pathlib import Path

CONFIGS = {
    'matched': dict(model='matched', q_layers=None, label='Matched-bottleneck classical'),
    'hybrid_L1': dict(model='hybrid', q_layers=1, label='Hybrid, 1 entangling layer'),
    'hybrid_L3': dict(model='hybrid', q_layers=3, label='Hybrid, 3 entangling layers (default)'),
    'classical': dict(model='classical', q_layers=None, label='Original classical CNN (128-unit head)'),
}
DEFAULT_CONFIGS = ['matched', 'hybrid_L1', 'hybrid_L3']
DEFAULT_SEEDS = [42, 43, 44]
METRICS = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc']


def run_dir(out, config, seed):
    return Path(out) / config / f'seed_{seed}'


def train_args(a, config, seed):
    c = CONFIGS[config]
    args = ['--manifest', a.manifest, '--out', str(run_dir(a.out, config, seed)), '--model', c['model'],
            '--epochs', str(a.epochs), '--batch-size', str(a.batch_size), '--lr', str(a.lr),
            '--patience', str(a.patience), '--seed', str(seed), '--split-seed', str(a.split_seed),
            '--data-kind', a.data_kind]
    if c['q_layers'] is not None:
        args += ['--q-layers', str(c['q_layers'])]
    return args


def run(a):
    from . import train
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plan = [(c, s) for c in a.configs for s in a.seeds]
    done = [(c, s) for c, s in plan if (run_dir(out, c, s) / 'metrics.json').is_file()]
    if done and not a.resume:
        sys.exit(f'error: {len(done)} run(s) already exist in {out}; use --resume to keep them and run only missing ones')
    spec = dict(configs={c: CONFIGS[c] for c in a.configs}, seeds=a.seeds, manifest=a.manifest, data_kind=a.data_kind,
                epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, patience=a.patience, split_seed=a.split_seed,
                threshold=0.5, selection='lowest validation loss', python=platform.python_version(),
                platform=platform.platform())
    (out / 'experiment.json').write_text(json.dumps(spec, indent=2))
    for config, seed in plan:
        if (config, seed) in done:
            print(f'[skip] {config} seed {seed} (already complete)', flush=True); continue
        print(f'[run ] {config} seed {seed}', flush=True)
        t0 = time.perf_counter()
        train.main(train_args(a, config, seed))
        print(f'[done] {config} seed {seed} in {time.perf_counter() - t0:.1f} s', flush=True)
    return aggregate(out)


def _summary(values):
    """Mean and sample standard deviation over defined values only."""
    vals = [v for v in values if v is not None]
    return dict(mean=statistics.fmean(vals) if vals else None,
                std=statistics.stdev(vals) if len(vals) > 1 else None,
                n_defined=len(vals), n_runs=len(values))


def collect_runs(out):
    rows = []
    for metrics_file in sorted(Path(out).glob('*/seed_*/metrics.json')):
        d = metrics_file.parent
        m = json.loads(metrics_file.read_text())
        r = json.loads((d / 'run.json').read_text()) if (d / 'run.json').is_file() else {}
        config = d.parent.name
        rows.append(dict(config=config, label=CONFIGS.get(config, {}).get('label', config), seed=m.get('seed'),
                         model=m.get('model'), q_layers=m.get('q_layers'), parameters=m.get('parameters'),
                         selected_epoch=m.get('selected_epoch'), epochs_run=m.get('epochs_run'),
                         train_seconds=m.get('train_seconds'), data_kind=m.get('data_kind'), n_test=m.get('n'),
                         confusion_matrix=m.get('confusion_matrix'), splits=r.get('splits'),
                         **{k: m.get(k) for k in METRICS}, run_dir=d.relative_to(out).as_posix()))
    return rows


def aggregate(out):
    out = Path(out)
    rows = collect_runs(out)
    if not rows:
        raise SystemExit(f'error: no completed runs found under {out}')
    order = [c for c in CONFIGS if any(r['config'] == c for r in rows)] + \
            sorted({r['config'] for r in rows} - set(CONFIGS))
    summary = []
    for c in order:
        rs = [r for r in rows if r['config'] == c]
        entry = dict(config=c, label=rs[0]['label'], model=rs[0]['model'], q_layers=rs[0]['q_layers'],
                     parameters=rs[0]['parameters'], seeds=[r['seed'] for r in rs], n_runs=len(rs),
                     data_kind=sorted({r['data_kind'] for r in rs}), n_test=sorted({r['n_test'] for r in rs}),
                     predicted_positive_per_run=[r['confusion_matrix'][0][1] + r['confusion_matrix'][1][1] for r in rs])
        for k in METRICS + ['train_seconds', 'selected_epoch']:
            entry[k] = _summary([r[k] for r in rs])
        summary.append(entry)
    note = ('Synthetic geometric fixtures only. With 4 test images per run, these figures show execution and '
            'preliminary behaviour, not clinical performance or statistically robust differences. Std is the '
            'sample standard deviation across seeds; undefined metrics are excluded and counted in n_defined. '
            'train_seconds is local wall-clock time on the recorded machine, not a benchmark.')
    (out / 'aggregate.json').write_text(json.dumps(dict(note=note, configs=summary), indent=2, allow_nan=False))
    with (out / 'runs.csv').open('w', newline='') as f:
        cols = ['config', 'seed', 'model', 'q_layers', 'parameters', 'selected_epoch', 'train_seconds'] + METRICS + ['confusion_matrix', 'data_kind', 'n_test']
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore', lineterminator='\n'); w.writeheader()
        for r in rows: w.writerow({**r, 'confusion_matrix': json.dumps(r['confusion_matrix'])})
    with (out / 'aggregate.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['config', 'parameters', 'n_runs'] + [f'{k}_{s}' for k in METRICS + ['train_seconds'] for s in ('mean', 'std', 'n_defined')])
        for e in summary:
            w.writerow([e['config'], e['parameters'], e['n_runs']] +
                       [('' if e[k][s] is None else (round(e[k][s], 4) if isinstance(e[k][s], float) else e[k][s]))
                        for k in METRICS + ['train_seconds'] for s in ('mean', 'std', 'n_defined')])
    print(json.dumps([{k: e[k] for k in ('config', 'n_runs')} | {m: e[m]['mean'] for m in ('accuracy', 'f1', 'roc_auc')} for e in summary], indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='train every configuration with every seed, then aggregate')
    r.add_argument('--manifest', required=True); r.add_argument('--out', required=True)
    r.add_argument('--data-kind', choices=['synthetic', 'prepared-ct'], required=True)
    r.add_argument('--configs', nargs='+', choices=list(CONFIGS), default=DEFAULT_CONFIGS)
    r.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    r.add_argument('--epochs', type=int, default=5); r.add_argument('--batch-size', type=int, default=4)
    r.add_argument('--lr', type=float, default=1e-4); r.add_argument('--patience', type=int, default=5)
    r.add_argument('--split-seed', type=int, default=42)
    r.add_argument('--resume', action='store_true', help='keep completed runs and only run missing ones')
    g = sub.add_parser('aggregate', help='rebuild aggregate.json / aggregate.csv / runs.csv from saved runs')
    g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if a.cmd == 'run':
        if len(set(a.seeds)) != len(a.seeds): p.error('seeds must be distinct')
        run(a)
    else:
        aggregate(a.out)
    return 0


if __name__ == '__main__': sys.exit(main())
