"""Tests for the end-semester additions: configurable circuit depth, the
matched-bottleneck comparator, checkpoint metadata, the experiment runner and
aggregation, and the app's input handling."""
import contextlib, io, json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from src.demo_data import create_demo


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


class DepthAndComparator(unittest.TestCase):
    def test_one_and_three_layer_construction(self):
        from src.hybrid_model import HybridModel
        for n, params in [(1, 479641), (3, 479657)]:
            m = HybridModel(n)
            self.assertEqual(tuple(m.quantum.q_layer.weights.shape), (n, 8))
            self.assertEqual(sum(p.numel() for p in m.parameters()), params)
            self.assertEqual(m.n_layers, n)
        self.assertEqual(HybridModel().n_layers, 3)  # default unchanged

    def test_invalid_depth_rejected(self):
        from src.quantum_layer import QuantumLayer
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                QuantumLayer(bad)

    def test_one_layer_gradients_finite_and_nonzero(self):
        from src.hybrid_model import HybridModel
        torch.manual_seed(0)
        m = HybridModel(1)
        y = m(torch.rand(2, 1, 128, 128))
        torch.nn.functional.binary_cross_entropy(y, torch.tensor([[0.], [1.]])).backward()
        for name in ['conv1.weight', 'fc1.weight', 'quantum.q_layer.weights', 'fc2.weight']:
            g = dict(m.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(g).all(), name); self.assertGreater(g.abs().sum().item(), 0, name)

    def test_matched_comparator_shapes_and_parameters(self):
        from src.matched_classical import MatchedBottleneckCNN
        from src.hybrid_model import HybridModel
        torch.manual_seed(0)
        m, h = MatchedBottleneckCNN(), HybridModel()
        x = torch.rand(3, 1, 128, 128)
        y, z = m(x), m.bottleneck(x)
        self.assertEqual(tuple(y.shape), (3, 1)); self.assertTrue(((y >= 0) & (y <= 1)).all())
        self.assertEqual(tuple(z.shape), (3, 8)); self.assertTrue(((z >= -1) & (z <= 1)).all())
        self.assertEqual(sum(p.numel() for p in m.parameters()), 479705)
        # same extractor and projection shapes as the hybrid
        for name in ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']:
            self.assertEqual(dict(m.named_parameters())[name].shape, dict(h.named_parameters())[name].shape)
        self.assertEqual(sum(p.numel() for p in m.mixer.parameters()), 72)

    def test_build_model_guards(self):
        from src.models import build_model
        with self.assertRaises(ValueError): build_model('matched', 1)
        with self.assertRaises(ValueError): build_model('quantum-cnn')


class CheckpointMetadata(unittest.TestCase):
    def save(self, path, model, name, q=None, include_q=True):
        ck = {'state_dict': model.state_dict(), 'model': name, 'epoch': 1, 'seed': 0, 'data_kind': 'synthetic'}
        if include_q: ck['q_layers'] = q
        torch.save(ck, path)

    def test_depth_roundtrip_rebuilds_correct_model(self):
        from src.hybrid_model import HybridModel
        from src.predict import load_model
        torch.manual_seed(1)
        x = torch.rand(1, 1, 128, 128)
        with tempfile.TemporaryDirectory() as d:
            for n in (1, 3):
                m = HybridModel(n).eval(); p = Path(d) / f'L{n}.pt'; self.save(p, m, 'hybrid', n)
                loaded, ck = load_model(p)
                self.assertEqual(ck['q_layers'], n); self.assertEqual(loaded.n_layers, n)
                with torch.no_grad(): self.assertTrue(torch.allclose(m(x), loaded(x)))
                self.assertFalse(any(q.requires_grad for q in loaded.parameters()))

    def test_legacy_checkpoint_without_depth_loads_as_three_layers(self):
        from src.hybrid_model import HybridModel
        from src.predict import load_model
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'old.pt'; self.save(p, HybridModel(3), 'hybrid', include_q=False)
            _, ck = load_model(p); self.assertEqual(ck['q_layers'], 3)

    def test_mislabelled_depth_gives_clear_error(self):
        from src.hybrid_model import HybridModel
        from src.predict import load_model
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.pt'; self.save(p, HybridModel(1), 'hybrid', 3)
            with self.assertRaisesRegex(ValueError, 'do not match'):
                load_model(p)

    def test_matched_checkpoint_roundtrip(self):
        from src.matched_classical import MatchedBottleneckCNN
        from src.predict import load_model
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'm.pt'; self.save(p, MatchedBottleneckCNN(), 'matched', None)
            m, ck = load_model(p); self.assertEqual(ck['model'], 'matched'); self.assertIsNone(ck['q_layers'])


class ExperimentRunner(unittest.TestCase):
    def test_aggregate_handles_undefined_metrics(self):
        from src.experiments import aggregate
        with tempfile.TemporaryDirectory() as d:
            for seed, (acc, prec) in zip((1, 2, 3), [(0.5, None), (0.75, 1.0), (1.0, 0.5)]):
                rd = Path(d) / 'matched' / f'seed_{seed}'; rd.mkdir(parents=True)
                (rd / 'metrics.json').write_text(json.dumps(dict(
                    n=4, accuracy=acc, precision=prec, recall=0.5, specificity=None, f1=0.5, roc_auc=1.0,
                    confusion_matrix=[[2, 0], [1, 1]], seed=seed, model='matched', q_layers=None,
                    parameters=479705, selected_epoch=2, train_seconds=1.0 + seed, data_kind='synthetic')))
            summary = quiet(aggregate, d)
            e = summary[0]
            self.assertAlmostEqual(e['accuracy']['mean'], 0.75); self.assertAlmostEqual(e['accuracy']['std'], 0.25)
            self.assertEqual(e['precision']['n_defined'], 2); self.assertAlmostEqual(e['precision']['mean'], 0.75)
            self.assertIsNone(e['specificity']['mean']); self.assertEqual(e['specificity']['n_defined'], 0)
            self.assertAlmostEqual(e['roc_auc']['std'], 0.0)
            self.assertTrue((Path(d) / 'aggregate.csv').is_file() and (Path(d) / 'runs.csv').is_file())
            json.loads((Path(d) / 'aggregate.json').read_text())  # strict JSON, no NaN

    def test_runner_end_to_end_and_refuses_rerun(self):
        from src.experiments import main
        with tempfile.TemporaryDirectory() as d:
            man = create_demo(Path(d) / 'data'); out = Path(d) / 'exp'
            args = ['run', '--manifest', str(man), '--out', str(out), '--data-kind', 'synthetic',
                    '--configs', 'matched', 'hybrid_L1', '--seeds', '1', '2', '--epochs', '1']
            quiet(main, args)
            agg = json.loads((out / 'aggregate.json').read_text())
            self.assertEqual([c['config'] for c in agg['configs']], ['matched', 'hybrid_L1'])
            self.assertTrue(all(c['n_runs'] == 2 for c in agg['configs']))
            run = json.loads((out / 'hybrid_L1' / 'seed_2' / 'run.json').read_text())
            self.assertEqual(run['config']['q_layers'], 1); self.assertEqual(run['config']['split_seed'], 42)
            self.assertIn('cpu_count', run['runtime'])
            m = json.loads((out / 'hybrid_L1' / 'seed_1' / 'metrics.json').read_text())
            self.assertGreater(m['train_seconds'], 0); self.assertEqual(m['parameters'], 479641)
            # same split for every run
            splits = {(out / c / f'seed_{s}' / 'split_manifest.csv').read_text() for c in ('matched', 'hybrid_L1') for s in (1, 2)}
            self.assertEqual(len(splits), 1)
            with self.assertRaises(SystemExit):
                quiet(main, args)
            quiet(main, args + ['--resume'])  # nothing left to run


class AppInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.hybrid_model import HybridModel
        cls.tmp = tempfile.TemporaryDirectory(); root = Path(cls.tmp.name)
        create_demo(root / 'demo_data')
        torch.manual_seed(0)
        (root / 'results' / 'hybrid').mkdir(parents=True)
        torch.save({'state_dict': HybridModel(1).state_dict(), 'model': 'hybrid', 'q_layers': 1, 'epoch': 2, 'seed': 0,
                    'data_kind': 'synthetic'}, root / 'results' / 'hybrid' / 'best.pt')
        (root / 'runs' / 'junk').mkdir(parents=True); (root / 'runs' / 'junk' / 'best.pt').write_bytes(b'not a checkpoint')
        cls.root = root

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def npy(self, arr):
        b = io.BytesIO(); np.save(b, arr, allow_pickle=True); return b.getvalue()

    def test_invalid_uploads_get_helpful_errors(self):
        from src.app_core import crop_from_upload
        cases = [('scan.png', b'\x89PNG\r\n', 'not a prepared .npy'), ('scan.dcm', b'DICM', 'DICOM'),
                 ('x.npy', b'', 'empty'), ('x.npy', b'garbage-bytes', 'not a readable'),
                 ('x.npy', self.npy(np.zeros((64, 64), np.float32)), '128x128'),
                 ('x.npy', self.npy(np.full((128, 128), 200, np.uint8)), 'divided by 255'),
                 ('x.npy', self.npy(np.array([{'a': 1}], dtype=object)), 'pickled'),
                 ('x.npy', self.npy(np.full((128, 128), np.nan, np.float32)), 'NaN')]
        for name, data, msg in cases:
            with self.assertRaisesRegex(ValueError, msg, msg=name):
                crop_from_upload(name, data)

    def test_discovery_and_deterministic_inference(self):
        from src import app_core as core
        from src.predict import load_model
        cks = core.discover_checkpoints(self.root)
        good = [c for c in cks if c['ok']]; bad = [c for c in cks if not c['ok']]
        self.assertEqual(len(good), 1); self.assertEqual(len(bad), 1)
        self.assertEqual(good[0]['q_layers'], 1); self.assertIn('synthetic', core.checkpoint_label(good[0]))
        rows = core.ensure_demo_samples(self.root)
        x = core.crop_from_upload('f.npy', Path(rows[23]['abs_path']).read_bytes())
        model, ck = load_model(good[0]['path'])
        a, b = core.run_inference(model, ck, x), core.run_inference(model, ck, x)
        self.assertEqual(a['score'], b['score']); self.assertEqual(a['threshold'], 0.5)
        self.assertEqual(a['predicted_class'], int(a['score'] >= 0.5)); self.assertEqual(a['data_kind'], 'synthetic')

    def test_demo_samples_regenerate_when_missing(self):
        from src import app_core as core
        with tempfile.TemporaryDirectory() as d:
            rows = core.ensure_demo_samples(d)
            self.assertEqual(len(rows), 24); self.assertTrue(Path(rows[0]['abs_path']).is_file())


@unittest.skipUnless(__import__('importlib').util.find_spec('streamlit'), 'streamlit not installed')
class AppSmoke(unittest.TestCase):
    def test_both_pages_render_without_exceptions(self):
        from streamlit.testing.v1 import AppTest
        app = Path(__file__).resolve().parents[1] / 'app' / 'streamlit_app.py'
        at = AppTest.from_file(str(app), default_timeout=120).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        at.sidebar.radio[0].set_value('Experiment results').run()
        self.assertFalse(at.exception, [e.value for e in at.exception])


if __name__ == '__main__': unittest.main()
