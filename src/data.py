"""Validated prepared-crop manifests; no implicit DICOM or label conversion."""
import csv
import hashlib
from pathlib import Path
import random
import numpy as np

SPLITS = ('train', 'val', 'test')

def load_manifest(path):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'path', 'patient_id', 'label'} <= set(reader.fieldnames or []):
            raise ValueError('Required columns: path,patient_id,label')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty manifest')
    seen = set()
    hashes = {}
    for row in rows:
        if not row['patient_id'].strip() or row['label'] not in ('0', '1'):
            raise ValueError('Patient IDs must be nonempty and labels must be 0 or 1')
        p = (path.parent / row['path']).resolve()
        if not p.is_file() or p.suffix.lower() != '.npy':
            raise ValueError(f'Expected existing .npy crop: {p}')
        if p in seen:
            raise ValueError('Repeated image path')
        seen.add(p)
        x = np.load(p, allow_pickle=False)
        if x.shape != (128,128) or not np.isfinite(x).all() or x.min()<0 or x.max()>1:
            raise ValueError('Crops must be finite 128x128 arrays normalized to [0,1]')
        digest = hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Duplicate crop contents detected')
        hashes[digest] = p
        row.update(path=str(p), label=int(row['label']), sha256=digest)
    supplied = [r.get('split', '').strip() for r in rows]
    if any(supplied):
        if not all(s in SPLITS for s in supplied):
            raise ValueError('Every row must have train, val or test when splits are supplied')
        for r,s in zip(rows,supplied): r['split']=s
        validate_splits(rows)
    return rows

def validate_splits(rows):
    patient_splits = {}
    for r in rows:
        s = r.get('split')
        if s not in SPLITS: raise ValueError('Missing or invalid split')
        patient_splits.setdefault(r['patient_id'], set()).add(s)
    if any(len(s)>1 for s in patient_splits.values()):
        raise ValueError('Patient leakage across splits')
    for s in SPLITS:
        labels = {r['label'] for r in rows if r['split']==s}
        if labels != {0,1}:
            raise ValueError(f'{s} must contain both classes; provide a reviewed split manifest')

def assign_splits(rows, seed=42):
    if all(r.get('split') in SPLITS for r in rows):
        validate_splits(rows)
        return rows
    patients = sorted({r['patient_id'] for r in rows})
    if len(patients)<7: raise ValueError('At least seven patients required for automatic splitting')
    random.Random(seed).shuffle(patients)
    n=len(patients); nval=max(1,round(n*.15)); ntest=max(1,round(n*.15))
    mapping={p:('test' if i<ntest else 'val' if i<ntest+nval else 'train') for i,p in enumerate(patients)}
    result=[dict(r,split=mapping[r['patient_id']]) for r in rows]
    validate_splits(result)
    return result
