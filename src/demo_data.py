"""Create labelled geometric fixtures, never medical images."""
import csv
from pathlib import Path
import numpy as np

def create_demo(destination):
    root=Path(destination); root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42); rows=[]
    yy,xx=np.mgrid[:128,:128]
    for i in range(24):
        label=i%2
        cx,cy=rng.integers(40,88,size=2)
        pattern=((xx-cx)**2+(yy-cy)**2 < (10+label*10)**2)
        x=np.clip(rng.normal(.15,.04,(128,128))+pattern*.6,0,1).astype('float32')
        name=f'fixture_{i:02}.npy'; np.save(root/name,x)
        rows.append(dict(path=name,patient_id=f'synthetic_{i:02}',label=label,split='train' if i<16 else 'val' if i<20 else 'test'))
    with (root/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','patient_id','label','split']);w.writeheader();w.writerows(rows)
    return root/'manifest.csv'
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',default='demo_data');a=p.parse_args();print(create_demo(a.out))
