"""Binary metrics with explicit undefined denominators."""
import numpy as np
from sklearn.metrics import roc_auc_score

def binary_metrics(labels, probabilities, threshold=.5):
    y=np.asarray(labels).reshape(-1); p=np.asarray(probabilities).reshape(-1)
    if len(y)==0 or y.shape!=p.shape or not np.isin(y,[0,1]).all(): raise ValueError('Invalid labels')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not 0<=threshold<=1: raise ValueError('Invalid probabilities or threshold')
    pred=p>=threshold
    tp=int(((y==1)&pred).sum()); tn=int(((y==0)&~pred).sum())
    fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum())
    div=lambda a,b:a/b if b else None
    return dict(n=len(y),threshold=threshold,accuracy=(tp+tn)/len(y),precision=div(tp,tp+fp),recall=div(tp,tp+fn),specificity=div(tn,tn+fp),f1=div(2*tp,2*tp+fp+fn),roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,confusion_matrix=[[tn,fp],[fn,tp]])
