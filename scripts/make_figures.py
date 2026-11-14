"""Regenerate the report/presentation figures from saved evidence.

Run from the project root:  python scripts/make_figures.py
Reads results/*/ (JSON/CSV/checkpoints) and demo_data/; writes docs/figures/.
Nothing here trains a model or changes results.
"""
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = ROOT / 'docs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
CLASSICAL, HYBRID = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})


def load(model):
    d = ROOT / 'results' / model
    return (json.loads((d / 'history.json').read_text()), json.loads((d / 'metrics.json').read_text()),
            list(csv.DictReader((d / 'predictions.csv').open())))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print(OUT / name)


def fixtures():
    rows = list(csv.DictReader((ROOT / 'demo_data' / 'manifest.csv').open()))
    pick = [r for r in rows if r['label'] == '0'][:2] + [r for r in rows if r['label'] == '1'][:2]
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
    for ax, r in zip(axes, pick):
        ax.imshow(np.load(ROOT / 'demo_data' / r['path']), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{r["path"].replace(".npy", "")}\nlabel {r["label"]} ({r["split"]})', fontsize=9)
        ax.axis('off')
    fig.suptitle('Synthetic geometric fixtures (not medical images): label 0 = radius-10 disc, label 1 = radius-20 disc',
                 fontsize=9.5, color=MUTED, y=0.06)
    save(fig, 'fig_fixtures.png')


def losses():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, model, color, title in [(axes[0], 'classical', CLASSICAL, 'Classical CNN'),
                                    (axes[1], 'hybrid', HYBRID, 'Hybrid CNN + quantum layer')]:
        h, m, _ = load(model); e = [x['epoch'] for x in h]
        ax.plot(e, [x['train_loss'] for x in h], '--o', color=color, lw=2, ms=6, alpha=.55, label='Train')
        ax.plot(e, [x['val_loss'] for x in h], '-s', color=color, lw=2, ms=6, label='Validation')
        ax.axvline(m['selected_epoch'], color=MUTED, ls=':', lw=1.2)
        ax.text(m['selected_epoch'], 0.722, f' selected\n epoch {m["selected_epoch"]}', color=MUTED, fontsize=8.5, va='top')
        ax.axhline(np.log(2), color=GRID, lw=1, zorder=0)
        ax.set(title=title, xlabel='Epoch', xticks=e)
        ax.grid(axis='y', color=GRID, lw=.8); ax.legend(frameon=False, loc='lower left')
    axes[0].set_ylabel('Binary cross-entropy')
    axes[1].text(1.0, 0.699, 'grey line: ln 2 = 0.693, the loss of a constant 0.5 output', color=MUTED, fontsize=8)
    fig.suptitle('Same y-axis for both models: the hybrid loss stays close to ln 2 throughout five epochs',
                 fontsize=10, color=INK, y=1.04)
    save(fig, 'fig_loss_shared_axis.png')


def scores():
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for k, (model, color, name) in enumerate([('classical', CLASSICAL, 'Classical'), ('hybrid', HYBRID, 'Hybrid')]):
        _, _, rows = load(model)
        for j, r in enumerate(rows):
            y = float(r['probability']); lab = int(r['label'])
            ax.scatter(j + (k - .5) * .18, y, s=70, color=color, marker='o' if lab else 's',
                       edgecolor='white', linewidth=1.5, zorder=3, label=f'{name}' if j == 0 else None)
    ax.axhline(.5, color=MUTED, ls='--', lw=1.2); ax.text(3.45, .5015, 'threshold 0.5', color=MUTED, fontsize=8.5, ha='right')
    _, _, rows = load('classical')
    ax.set_xticks(range(4), [f'{r["patient_id"].replace("synthetic_", "fixture ")}\n(label {r["label"]})' for r in rows])
    ax.set_ylabel('Sigmoid score for class 1'); ax.grid(axis='y', color=GRID, lw=.8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls='', marker='o', ms=8, color=CLASSICAL, label='Classical'),
         Line2D([], [], ls='', marker='o', ms=8, color=HYBRID, label='Hybrid'),
         Line2D([], [], ls='', marker='o', ms=8, color=MUTED, label='true label 1 (circle)'),
         Line2D([], [], ls='', marker='s', ms=8, color=MUTED, label='true label 0 (square)')]
    ax.legend(handles=h, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=4)
    ax.set_title('All four held-out test scores: ranking is perfect for both models, the 0.5 cut is not', fontsize=10)
    save(fig, 'fig_test_scores.png')


def hybrid_signal():
    from src.diagnose import hybrid_signals
    from src.predict import load_model
    from src.data import load_manifest, load_crop
    model, _ = load_model(ROOT / 'results' / 'hybrid' / 'best.pt')
    rows = load_manifest(ROOT / 'demo_data' / 'manifest.csv')
    x = torch.stack([torch.from_numpy(load_crop(r['path'])) for r in rows]).unsqueeze(1)
    with torch.no_grad():
        _, q, _ = hybrid_signals(model, x)
    q = q.numpy()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i in range(8):
        ax.scatter(np.full(len(q), i) + np.linspace(-.18, .18, len(q)), q[:, i], s=14, color=HYBRID, alpha=.8)
    ax.set(xticks=range(8), xticklabels=[f'qubit {i}' for i in range(8)], ylim=(-1.05, 1.05),
           ylabel='Pauli-Z expectation value')
    ax.axhline(0, color=GRID, lw=1); ax.grid(axis='y', color=GRID, lw=.8)
    ax.set_title('Trained hybrid: 24 fixtures produce almost identical expectation values (possible range -1 to +1)',
                 fontsize=10)
    save(fig, 'fig_hybrid_expvals.png')


def circuit():
    import pennylane as qml
    from src.quantum_layer import quantum_circuit
    tape_fn = qml.draw_mpl(quantum_circuit, level='device', decimals=None, style='black_white')
    fig, _ = tape_fn(np.zeros(8), np.zeros((3, 8)))
    save(fig, 'fig_circuit_gates.png')


def architecture():
    blocks = [('Input', '1 x 128 x 128\nprepared crop', '#dce9f8'),
              ('Conv block 1', 'Conv2d(1,32,3)\nReLU, MaxPool2\n32 x 63 x 63', '#dce9f8'),
              ('Conv block 2', 'Conv2d(32,64,3)\nReLU, MaxPool2\n64 x 30 x 30', '#dce9f8'),
              ('Projection', 'Flatten 57,600\nLinear -> 8\n(no activation)', '#dce9f8'),
              ('Quantum layer', '8 qubits\nRX angle encoding\n3 entangling layers\n(24 trainable angles)', '#fde3d7'),
              ('Readout', '8 Pauli-Z\nexpectation\nvalues', '#fde3d7'),
              ('Classifier', 'Linear(8,1)\nsigmoid\nscore in [0,1]', '#dce9f8')]
    fig, ax = plt.subplots(figsize=(13, 3.1)); ax.set_xlim(0, 7); ax.set_ylim(0, 1); ax.axis('off')
    for i, (t, body, c) in enumerate(blocks):
        ax.add_patch(FancyBboxPatch((i + .06, .1), .84, .72, boxstyle='round,pad=0.02,rounding_size=0.05',
                                    fc=c, ec=MUTED, lw=1))
        ax.text(i + .48, .74, t, ha='center', va='top', fontsize=9.5, fontweight='bold', color=INK)
        ax.text(i + .48, .40, body, ha='center', va='center', fontsize=8, color=INK, linespacing=1.3)
        if i < len(blocks) - 1:
            ax.annotate('', xy=(i + 1.06, .46), xytext=(i + .92, .46), arrowprops=dict(arrowstyle='->', color=MUTED))
    ax.text(3.5, .96, 'Hybrid model as implemented in src/hybrid_model.py and src/quantum_layer.py '
            '(simulated with PennyLane default.qubit, backpropagation)', ha='center', fontsize=9.5, color=MUTED)
    ax.text(3.5, .0, 'Classical comparator: same two conv blocks, then Linear(57,600 -> 128) + ReLU and Linear(128 -> 1) + sigmoid',
            ha='center', fontsize=9, color=MUTED)
    save(fig, 'fig_architecture.png')


def experiment_curves(name='endsem_synthetic'):
    exp = ROOT / 'experiments' / name
    if not (exp / 'aggregate.json').is_file():
        print('skip experiment_curves: no', exp); return
    configs = [c['config'] for c in json.loads((exp / 'aggregate.json').read_text())['configs']]
    titles = {'matched': 'Matched classical', 'hybrid_L1': 'Hybrid, 1 layer', 'hybrid_L3': 'Hybrid, 3 layers', 'classical': 'Original classical'}
    colors = {'matched': '#1baf7a', 'hybrid_L1': '#6250d6', 'hybrid_L3': HYBRID, 'classical': CLASSICAL}
    fig, axes = plt.subplots(1, len(configs), figsize=(3.1 * len(configs), 3.2), sharey=True)
    for ax, c in zip(axes, configs):
        for k, d in enumerate(sorted((exp / c).glob('seed_*'))):
            h = json.loads((d / 'history.json').read_text()); m = json.loads((d / 'metrics.json').read_text())
            ax.plot([x['epoch'] for x in h], [x['val_loss'] for x in h], '-o', ms=4, lw=1.6, color=colors.get(c, MUTED),
                    alpha=(1.0, .65, .4)[k % 3], label=f'seed {m["seed"]}')
        ax.axhline(np.log(2), color=GRID, lw=1, zorder=0)
        ax.set(title=titles.get(c, c), xlabel='Epoch', xticks=range(1, 6)); ax.grid(axis='y', color=GRID, lw=.8)
        ax.legend(frameon=False, fontsize=8, loc='lower left')
    axes[0].set_ylabel('Validation BCE')
    fig.suptitle('Validation loss per seed (same 16/4/4 synthetic split; grey line = ln 2)', fontsize=10, y=1.02)
    save(fig, 'fig_seed_val_loss.png')


if __name__ == '__main__':
    fixtures(); losses(); scores(); hybrid_signal(); circuit(); architecture(); experiment_curves()
