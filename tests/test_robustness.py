"""Tests added in the November 2026 refinement: realistic invalid inputs,
portable evidence files, split/seed separation and the CLI tools."""
import contextlib, csv, io, json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from src.data import load_manifest, assign_splits
from src.demo_data import create_demo


def manifest_with(d, old, new):
    m = create_demo(d)
    m.write_text(m.read_text().replace(old, new))
    return m


class ManifestRobustness(unittest.TestCase):
    def test_whitespace_patient_id_cannot_hide_leakage(self):
        # Before the fix, "synthetic_01 " counted as a different patient, so a
        # test crop could silently share a patient with a training crop.
        with tempfile.TemporaryDirectory() as d:
            m = manifest_with(d, 'fixture_23.npy,synthetic_23', 'fixture_23.npy,synthetic_01 ')
            with self.assertRaisesRegex(ValueError, 'leakage'):
                load_manifest(m)

    def test_case_variant_patient_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            m = manifest_with(d, 'fixture_23.npy,synthetic_23', 'fixture_23.npy,SYNTHETIC_01')
            with self.assertRaisesRegex(ValueError, 'differs only by case'):
                load_manifest(m)

    def test_missing_fields_give_line_number(self):
        with tempfile.TemporaryDirectory() as d:
            m = create_demo(d)
            lines = m.read_text().splitlines(); lines[1] = 'fixture_00.npy'
            m.write_text('\n'.join(lines) + '\n')
            with self.assertRaisesRegex(ValueError, 'line 2'):
                load_manifest(m)

    def test_spreadsheet_style_values_are_normalised(self):
        with tempfile.TemporaryDirectory() as d:
            m = create_demo(d)
            text = m.read_text().replace(',train', ', Train ').replace(',1,', ', 1 ,')
            m.write_text(text)
            rows = load_manifest(m)
            self.assertEqual({r['split'] for r in rows}, {'train', 'val', 'test'})
            self.assertEqual(sum(r['label'] for r in rows), 12)

    def test_eight_bit_crop_rejected_with_hint(self):
        with tempfile.TemporaryDirectory() as d:
            m = create_demo(d); np.save(Path(d) / 'fixture_00.npy', np.full((128, 128), 200, np.uint8))
            with self.assertRaisesRegex(ValueError, 'divided by 255'):
                load_manifest(m)

    def test_non_npy_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            m = manifest_with(d, 'fixture_00.npy', 'fixture_00.png')
            (Path(d) / 'fixture_00.png').write_bytes(b'\x89PNG')
            with self.assertRaisesRegex(ValueError, r'\.npy'):
                load_manifest(m)

    def test_split_seed_controls_fallback_split(self):
        with tempfile.TemporaryDirectory() as d:
            rows = load_manifest(create_demo(d))
            for r in rows: r.pop('split')
            got = {}
            for seed in range(60):
                try: got[seed] = tuple(r['split'] for r in assign_splits(rows, seed))
                except ValueError: pass
            self.assertGreater(len(set(got.values())), 1)  # different seeds, different splits
            s = next(iter(got)); self.assertEqual(got[s], tuple(r['split'] for r in assign_splits(rows, s)))


class TrainingEvidence(unittest.TestCase):
    def run_train(self, out, *extra):
        from src.train import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['--manifest', str(self.manifest), '--out', str(out), '--model', 'classical',
                  '--epochs', '1', '--data-kind', 'synthetic', *extra])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.manifest = create_demo(self.root / 'data')

    def tearDown(self): self.tmp.cleanup()

    def test_evidence_files_have_no_absolute_paths(self):
        out = self.root / 'run'; self.run_train(out)
        for name in ['split_manifest.csv', 'predictions.csv', 'run.json']:
            text = (out / name).read_text()
            self.assertNotIn(str(self.root), text, f'{name} leaks an absolute local path')
        with (out / 'predictions.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]['path'], 'fixture_20.npy')
        for name in ['best.pt', 'metrics.json', 'run.json', 'history.json', 'training_curves.png',
                     'confusion_matrix.png', 'roc_curve.png']:
            self.assertTrue((out / name).is_file(), name)
        run = json.loads((out / 'run.json').read_text())
        self.assertEqual(run['split_source'], 'manifest')
        self.assertEqual(run['config']['split_seed'], 42)

    def test_refuses_nonempty_output_directory(self):
        out = self.root / 'run'; out.mkdir(); (out / 'keep.txt').write_text('previous run')
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_train(out)
        self.assertEqual((out / 'keep.txt').read_text(), 'previous run')


class PredictCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.hybrid_model import HybridModel
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        create_demo(cls.root)
        torch.manual_seed(0)
        cls.ck = cls.root / 'ck.pt'
        torch.save({'state_dict': HybridModel().state_dict(), 'model': 'hybrid', 'epoch': 1, 'seed': 0,
                    'data_kind': 'synthetic'}, cls.ck)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def call(self, ck, image):
        from src.predict import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(['--checkpoint', str(ck), '--image', str(image)])
        return code, out.getvalue(), err.getvalue()

    def test_valid_prediction_is_labelled_uncalibrated(self):
        code, out, _ = self.call(self.ck, self.root / 'fixture_23.npy')
        self.assertEqual(code, 0); r = json.loads(out)
        self.assertTrue(0 <= r['score_class_1'] <= 1)
        self.assertFalse(r['calibrated_probability']); self.assertFalse(r['clinical_validation'])
        self.assertEqual(r['training_data_kind'], 'synthetic')

    def test_clear_errors_instead_of_tracebacks(self):
        bare = self.root / 'bare.pt'
        torch.save(torch.load(self.ck, weights_only=True)['state_dict'], bare)
        cases = [(self.ck, self.root / 'missing.npy', 'not found'),
                 (self.ck, self.root / 'scan.png', '.npy'),
                 (bare, self.root / 'fixture_23.npy', 'not a checkpoint produced by src.train'),
                 (self.root / 'nope.pt', self.root / 'fixture_23.npy', 'checkpoint not found')]
        for ck, image, msg in cases:
            code, out, err = self.call(ck, image)
            self.assertEqual(code, 2, err); self.assertIn(msg, err); self.assertEqual(out, '')

    def test_diagnose_reports_signal_spread(self):
        from src.diagnose import summarise
        s = summarise(self.ck, self.root / 'manifest.csv')
        self.assertEqual(s['images'], 24)
        self.assertGreaterEqual(s['expval_std_across_images_mean'], 0)
        self.assertLessEqual(s['score_min'], s['score_max'])


if __name__ == '__main__': unittest.main()
