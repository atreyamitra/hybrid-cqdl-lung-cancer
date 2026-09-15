import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def save_plots(out,history,y,p,metrics):
    note='Synthetic fixture' if metrics['data_kind']=='synthetic' else 'Prepared crop evaluation'
    fig,ax=plt.subplots();ax.plot([h['epoch'] for h in history],[h['train_loss'] for h in history],label='Train');ax.plot([h['epoch'] for h in history],[h['val_loss'] for h in history],label='Validation');ax.set(xlabel='Epoch',ylabel='Binary cross entropy',title=f'{note}: {metrics["model"]}');ax.legend();fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots();cm=metrics['confusion_matrix'];ax.imshow(cm,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted label',ylabel='True label',title=f'{note}: confusion matrix')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i][j]),ha='center',va='center',color='white' if cm[i][j]>max(map(max,cm))/2 else 'black')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)
    if metrics['roc_auc'] is not None:
        fpr,tpr,_=roc_curve(y,p);fig,ax=plt.subplots();ax.plot(fpr,tpr,label=f'AUC {metrics["roc_auc"]:.3f}');ax.plot([0,1],[0,1],':');ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{note}: ROC');ax.legend();fig.tight_layout();fig.savefig(out/'roc_curve.png',dpi=160);plt.close(fig)
