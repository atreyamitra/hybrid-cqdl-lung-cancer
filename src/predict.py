"""Single prepared crop inference with checkpoint provenance."""
import argparse,json
import numpy as np
import torch
from .hybrid_model import HybridModel
from .classical_cnn import ClassicalCNN

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    m=HybridModel() if ck['model']=='hybrid' else ClassicalCNN();m.load_state_dict(ck['state_dict']);m.eval()
    x=np.load(a.image,allow_pickle=False)
    if x.shape!=(128,128) or not np.isfinite(x).all() or x.min()<0 or x.max()>1:raise ValueError('Expected normalized finite 128x128 crop')
    with torch.no_grad():pr=m(torch.tensor(x,dtype=torch.float32)[None,None]).item()
    print(json.dumps(dict(probability_class_1=pr,predicted_label=int(pr>=.5),training_data_kind=ck['data_kind'],clinical_validation=False),indent=2))
if __name__=='__main__':main()
