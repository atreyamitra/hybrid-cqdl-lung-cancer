"""Single prepared-crop inference with checkpoint provenance.

The score is the model's sigmoid output. It is not a calibrated cancer
probability, and checkpoints trained on synthetic fixtures say so in the output.
"""
import argparse, json, sys
from pathlib import Path
import torch
from .data import load_crop
from .models import MODEL_TYPES, build_model, describe

THRESHOLD = 0.5
REQUIRED_KEYS = {'state_dict', 'model', 'data_kind'}


def load_model(checkpoint_path):
    """Load a checkpoint written by ``src.train`` (weights only, no pickle)."""
    p = Path(checkpoint_path)
    if not p.is_file():
        raise ValueError(f'{p}: checkpoint not found')
    try:
        ck = torch.load(p, map_location='cpu', weights_only=True)
    except Exception as e:
        raise ValueError(f'{p}: not a readable PyTorch weights-only checkpoint ({e.__class__.__name__})') from None
    if not isinstance(ck, dict) or not REQUIRED_KEYS <= set(ck):
        raise ValueError(f'{p}: not a checkpoint produced by src.train (expected keys {sorted(REQUIRED_KEYS)})')
    if ck['model'] not in MODEL_TYPES:
        raise ValueError(f'{p}: unknown model type {ck["model"]!r}')
    # Checkpoints written before depth became configurable have no q_layers
    # key; they were all trained with the original three layers.
    q = ck.get('q_layers', 3 if ck['model'] == 'hybrid' else None)
    ck = dict(ck, q_layers=q)
    try:
        m = build_model(ck['model'], q)
        m.load_state_dict(ck['state_dict'])
    except (RuntimeError, ValueError):
        raise ValueError(f'{p}: weights do not match {describe(ck["model"], q)}') from None
    for param in m.parameters():
        param.requires_grad_(False)
    return m.eval(), ck


def score_array(model, x):
    """Deterministic single-crop inference: eval mode, no autograd graph."""
    model.eval()
    with torch.inference_mode():
        return float(model(torch.from_numpy(x)[None, None]).item())


def predict(checkpoint_path, image_path):
    model, ck = load_model(checkpoint_path)
    x = load_crop(image_path)
    score = score_array(model, x)
    return dict(model=ck['model'], q_layers=ck['q_layers'], checkpoint_epoch=ck.get('epoch'), score_class_1=score,
                predicted_label=int(score >= THRESHOLD), threshold=THRESHOLD,
                training_data_kind=ck['data_kind'], calibrated_probability=False, clinical_validation=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--image', required=True)
    a = p.parse_args(argv)
    try:
        result = predict(a.checkpoint, a.image)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__': sys.exit(main())
