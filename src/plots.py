import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def save_plots(out, history, y, p, metrics):
    note = 'Synthetic fixtures' if metrics['data_kind'] == 'synthetic' else 'Prepared crops'
    q = metrics.get('q_layers')
    name = {'hybrid': f'Hybrid CNN + quantum layer{f" (L={q})" if q else ""}', 'classical': 'Classical CNN',
            'matched': 'Matched-bottleneck classical'}.get(metrics['model'], metrics['model'])
    epochs = [h['epoch'] for h in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h['train_loss'] for h in history], 'o-', label='Train')
    ax.plot(epochs, [h['val_loss'] for h in history], 's-', label='Validation')
    ax.axvline(metrics['selected_epoch'], color='0.5', ls=':', label=f'Selected epoch {metrics["selected_epoch"]}')
    ax.set(xlabel='Epoch', ylabel='Binary cross-entropy', title=f'{name}: loss ({note})', xticks=epochs)
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    cm = metrics['confusion_matrix']
    ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['0', '1'], yticklabels=['0', '1'],
           xlabel='Predicted label', ylabel='True label', title=f'{name}\nconfusion matrix (n={metrics["n"]}, t=0.5)')
    top = max(map(max, cm)) or 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i][j] > top / 2 else 'black')
    fig.tight_layout(); fig.savefig(out / 'confusion_matrix.png', dpi=160); plt.close(fig)

    if metrics['roc_auc'] is not None:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.plot(fpr, tpr, 'o-', label=f'AUC {metrics["roc_auc"]:.3f} (n={metrics["n"]})')
        ax.plot([0, 1], [0, 1], ':', color='0.5')
        ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{name}\nROC ({note})')
        ax.legend(loc='lower right'); fig.tight_layout(); fig.savefig(out / 'roc_curve.png', dpi=160); plt.close(fig)
