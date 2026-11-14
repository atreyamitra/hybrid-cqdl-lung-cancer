"""Validated prepared-crop manifests; no implicit DICOM or label conversion.

The manifest is a CSV with columns ``path,patient_id,label`` and an optional
``split`` column (train/val/test). Every check fails fast with a message that
names the offending CSV line, because silent data errors (especially patient
leakage) produce attractive but meaningless metrics.
"""
import csv
import hashlib
from pathlib import Path
import random
import numpy as np

SPLITS = ('train', 'val', 'test')
CROP_SHAPE = (128, 128)


def validate_crop(x, source='crop'):
    """Return ``x`` as float32 after checking the prepared-crop contract.

    Shared by the training loader and the prediction CLI so both accept and
    reject exactly the same inputs.
    """
    if not isinstance(x, np.ndarray) or x.dtype.kind not in 'fiub':
        raise ValueError(f'{source}: expected a numeric NumPy array')
    if x.shape != CROP_SHAPE:
        raise ValueError(f'{source}: crops must be finite 128x128 arrays normalized to [0,1]; got shape {x.shape}')
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f'{source}: crops must be finite 128x128 arrays normalized to [0,1]; found NaN or infinity')
    if x.min() < 0 or x.max() > 1:
        raise ValueError(f'{source}: crops must be finite 128x128 arrays normalized to [0,1]; '
                         f'value range is [{x.min():.3g}, {x.max():.3g}] (8-bit images must be divided by 255)')
    return x


def load_crop(path):
    """Load one .npy crop without pickle support and validate it."""
    p = Path(path)
    if p.suffix.lower() != '.npy':
        raise ValueError(f'{p}: expected a prepared .npy crop, not {p.suffix or "a file without extension"}')
    if not p.is_file():
        raise ValueError(f'{p}: file not found')
    try:
        x = np.load(p, allow_pickle=False)
    except ValueError as e:
        raise ValueError(f'{p}: not a plain numeric .npy array ({e.__class__.__name__})') from None
    return validate_crop(x, str(p))


def _field(row, name, line):
    value = row.get(name)
    if value is None:
        raise ValueError(f'Manifest line {line}: missing value for "{name}"')
    return value.strip()


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = [c.strip() for c in (reader.fieldnames or [])]
        if not {'path', 'patient_id', 'label'} <= set(fields):
            raise ValueError('Required columns: path,patient_id,label')
        reader.fieldnames = fields
        raw = [r for r in reader if any((v or '').strip() for v in r.values() if isinstance(v, str))]
    if not raw:
        raise ValueError('Empty manifest')
    has_split = 'split' in fields
    rows, seen, hashes, id_forms = [], set(), {}, {}
    for i, r in enumerate(raw):
        line = i + 2  # header is line 1
        rel = _field(r, 'path', line)
        pid = _field(r, 'patient_id', line)
        label = _field(r, 'label', line)
        if not pid:
            raise ValueError(f'Manifest line {line}: patient IDs must be nonempty')
        if label not in ('0', '1'):
            raise ValueError(f'Manifest line {line}: labels must be 0 or 1, got "{label}"')
        # IDs that differ only by letter case are almost always the same patient
        # typed twice; refusing them prevents hidden cross-split leakage.
        canonical = pid.casefold()
        if id_forms.setdefault(canonical, pid) != pid:
            raise ValueError(f'Manifest line {line}: patient ID "{pid}" differs only by case from '
                             f'"{id_forms[canonical]}"; use one spelling per patient')
        p = (path.parent / rel).resolve()
        if p in seen:
            raise ValueError(f'Manifest line {line}: repeated image path {rel}')
        seen.add(p)
        try:
            x = load_crop(p)
        except ValueError as e:
            raise ValueError(f'Manifest line {line}: {e}') from None
        digest = hashlib.sha256(x.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f'Manifest line {line}: Duplicate crop contents detected (same as {hashes[digest].name})')
        hashes[digest] = p
        row = dict(path=str(p), relpath=rel, patient_id=pid, label=int(label), sha256=digest)
        if has_split:
            row['split'] = (r.get('split') or '').strip().lower()
        rows.append(row)
    supplied = [r.get('split', '') for r in rows]
    if any(supplied):
        bad = [i + 2 for i, s in enumerate(supplied) if s not in SPLITS]
        if bad:
            raise ValueError(f'Every row must have train, val or test when splits are supplied (check lines {bad[:5]})')
        validate_splits(rows)
    return rows


def validate_splits(rows):
    patient_splits = {}
    for r in rows:
        s = r.get('split')
        if s not in SPLITS:
            raise ValueError('Missing or invalid split')
        patient_splits.setdefault(r['patient_id'], set()).add(s)
    leaked = sorted(p for p, s in patient_splits.items() if len(s) > 1)
    if leaked:
        raise ValueError(f'Patient leakage across splits: {leaked[:5]}')
    for s in SPLITS:
        labels = {r['label'] for r in rows if r['split'] == s}
        if labels != {0, 1}:
            raise ValueError(f'{s} must contain both classes; provide a reviewed split manifest')


def assign_splits(rows, seed=42):
    """Use reviewed splits if present, else a seeded ~70/15/15 patient split.

    The fallback is not stratified. If it leaves a class missing it fails
    instead of searching for a convenient seed.
    """
    if all(r.get('split') in SPLITS for r in rows):
        validate_splits(rows)
        return rows
    patients = sorted({r['patient_id'] for r in rows})
    if len(patients) < 7:
        raise ValueError('At least seven patients required for automatic splitting')
    random.Random(seed).shuffle(patients)
    n = len(patients)
    nval = max(1, round(n * .15))
    ntest = max(1, round(n * .15))
    mapping = {p: ('test' if i < ntest else 'val' if i < ntest + nval else 'train') for i, p in enumerate(patients)}
    result = [dict(r, split=mapping[r['patient_id']]) for r in rows]
    validate_splits(result)
    return result
