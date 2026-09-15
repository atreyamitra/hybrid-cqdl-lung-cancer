"""Reproducible CPU training and held-out evaluation for prepared crops."""
import argparse, csv, hashlib, json, platform, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data import load_manifest, assign_splits, SPLITS
from .metrics import binary_metrics
from .hybrid_model import HybridModel
from .classical_cnn import ClassicalCNN

class Crops(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return torch.from_numpy(np.load(r['path'],allow_pickle=False).astype('float32')).unsqueeze(0),torch.tensor([r['label']],dtype=torch.float32)

def evaluate(model, loader):
    model.eval(); ys=[]; ps=[]; losses=[]
    with torch.no_grad():
        for x,y in loader:
            p=model(x); losses.append((torch.nn.functional.binary_cross_entropy(p,y).item(),len(y)))
            ys.extend(y.flatten().tolist()); ps.extend(p.flatten().tolist())
    return sum(a*b for a,b in losses)/sum(b for _,b in losses),ys,ps

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model',choices=['hybrid','classical'],default='hybrid')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--data-kind',choices=['synthetic','prepared-ct'],required=True)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.patience)<=0 or a.lr<=0:p.error('Training settings must be positive')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty to protect previous runs')
    rows=assign_splits(load_manifest(a.manifest),a.seed)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=True)
    loaders={s:DataLoader(Crops([r for r in rows if r['split']==s]),batch_size=a.batch_size,shuffle=s=='train',generator=torch.Generator().manual_seed(a.seed)) for s in SPLITS}
    model=HybridModel() if a.model=='hybrid' else ClassicalCNN()
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    history=[];best=float('inf');stale=0;start=time.perf_counter()
    for epoch in range(1,a.epochs+1):
        model.train();loss_sum=0;n=0
        for x,y in loaders['train']:
            optimizer.zero_grad();prob=model(x);loss=torch.nn.functional.binary_cross_entropy(prob,y)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();optimizer.step();loss_sum+=loss.item()*len(y);n+=len(y)
        val_loss,_,_=evaluate(model,loaders['val'])
        history.append(dict(epoch=epoch,train_loss=loss_sum/n,val_loss=val_loss))
        print(history[-1],flush=True)
        if val_loss<best:
            best=val_loss;stale=0
            torch.save({'state_dict':model.state_dict(),'model':a.model,'epoch':epoch,'seed':a.seed,'data_kind':a.data_kind},out/'best.pt')
        else:stale+=1
        if stale>=a.patience:break
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    _,y,prob=evaluate(model,loaders['test'])
    result=binary_metrics(y,prob)
    result.update(data_kind=a.data_kind,model=a.model,selected_epoch=checkpoint['epoch'],parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.perf_counter()-start,clinical_validation=False)
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (out/'history.json').write_text(json.dumps(history,indent=2))
    import pennylane
    metadata=dict(config=vars(a),python=platform.python_version(),torch=torch.__version__,pennylane=pennylane.__version__,numpy=np.__version__,manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),splits={s:dict(images=sum(r['split']==s for r in rows),patients=len({r['patient_id'] for r in rows if r['split']==s})) for s in SPLITS})
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    with (out/'split_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','patient_id','label','split','sha256'],extrasaction='ignore');w.writeheader();w.writerows(rows)
    with (out/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['patient_id','path','label','probability'])
        for r,pr in zip([r for r in rows if r['split']=='test'],prob):w.writerow([r['patient_id'],r['path'],r['label'],pr])
    from .plots import save_plots
    save_plots(out,history,y,prob,result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
