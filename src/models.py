"""Single place that maps a model name (+ quantum depth) to an nn.Module.

Used by training, prediction, the experiment runner and the app, so a checkpoint
always rebuilds the architecture it was trained with.
"""
from .classical_cnn import ClassicalCNN
from .hybrid_model import HybridModel
from .matched_classical import MatchedBottleneckCNN
from .quantum_layer import DEFAULT_LAYERS

MODEL_TYPES = ('hybrid', 'classical', 'matched')
DESCRIPTIONS = {
    'hybrid': 'Hybrid CNN + 8-qubit variational layer',
    'classical': 'Original classical CNN (128-unit head)',
    'matched': 'Matched-bottleneck classical CNN (8 -> Linear(8,8)+tanh -> 1)',
}


def build_model(model, q_layers=None):
    if model not in MODEL_TYPES:
        raise ValueError(f'unknown model type {model!r}; expected one of {MODEL_TYPES}')
    if model == 'hybrid':
        return HybridModel(DEFAULT_LAYERS if q_layers is None else q_layers)
    if q_layers not in (None,):
        raise ValueError('q_layers applies only to the hybrid model')
    return ClassicalCNN() if model == 'classical' else MatchedBottleneckCNN()


def describe(model, q_layers=None):
    if model == 'hybrid':
        n = DEFAULT_LAYERS if q_layers is None else q_layers
        return f'{DESCRIPTIONS[model]}, {n} entangling layer{"s" if n != 1 else ""}'
    return DESCRIPTIONS[model]


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
