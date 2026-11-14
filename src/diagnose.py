"""Inspect the signal reaching and leaving the quantum layer of a hybrid checkpoint.

Reports, for every crop in a manifest, the eight encoded rotation angles (the
outputs of the 57,600 -> 8 projection), the eight Pauli-Z expectation values and
the final score. Small spread across images means the classifier receives
almost the same eight numbers for every input, which explains scores that sit
close together around 0.5. This is a diagnostic, not a performance metric.
"""
import argparse, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from .data import load_manifest
from .predict import load_model
from .data import load_crop


def hybrid_signals(model, x):
    h = model.pool(F.relu(model.conv1(x)))
    h = model.pool(F.relu(model.conv2(h)))
    angles = model.fc1(torch.flatten(h, 1))
    expvals = model.quantum(angles)
    return angles, expvals, torch.sigmoid(model.fc2(expvals))


def summarise(checkpoint, manifest):
    model, ck = load_model(checkpoint)
    if ck['model'] != 'hybrid':
        raise ValueError('diagnose expects a hybrid checkpoint')
    rows = load_manifest(manifest)
    x = torch.stack([torch.from_numpy(load_crop(r['path'])) for r in rows]).unsqueeze(1)
    with torch.no_grad():
        angles, expvals, scores = hybrid_signals(model, x)
    r = lambda v: round(float(v.detach()), 4)
    return dict(
        checkpoint_epoch=ck.get('epoch'), images=len(rows), training_data_kind=ck['data_kind'],
        angle_abs_mean=r(angles.abs().mean()), angle_abs_max=r(angles.abs().max()),
        expval_min=r(expvals.min()), expval_max=r(expvals.max()),
        expval_std_across_images_mean=r(expvals.std(0).mean()),
        classifier_weight_abs_mean=r(model.fc2.weight.abs().mean()),
        score_min=r(scores.min()), score_max=r(scores.max()),
        note='Spread across images, not accuracy. Synthetic checkpoints say nothing about CT behaviour.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--manifest', required=True)
    a = p.parse_args(argv)
    try:
        print(json.dumps(summarise(a.checkpoint, a.manifest), indent=2))
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr); return 2
    return 0


if __name__ == '__main__': sys.exit(main())
