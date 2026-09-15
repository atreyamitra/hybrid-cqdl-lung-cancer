import unittest,tempfile
from pathlib import Path
import numpy as np
from src.metrics import binary_metrics
from src.data import load_manifest,assign_splits,validate_splits
from src.demo_data import create_demo

class DataTests(unittest.TestCase):
    def test_metrics(self):
        m=binary_metrics([0,0,1,1],[.1,.8,.7,.9]);self.assertEqual(m['confusion_matrix'],[[1,1],[0,2]]);self.assertEqual(m['accuracy'],.75);self.assertAlmostEqual(m['f1'],.8);self.assertAlmostEqual(m['roc_auc'],.75)
    def test_undefined(self):
        m=binary_metrics([0,0],[.1,.2]);self.assertIsNone(m['recall']);self.assertIsNone(m['roc_auc'])
    def test_bad_probabilities(self):
        with self.assertRaises(ValueError):binary_metrics([1],[float('nan')])
    def test_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            rows=load_manifest(create_demo(d));self.assertEqual(len(assign_splits(rows)),24)
    def test_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            rows=load_manifest(create_demo(d));rows[-1]['patient_id']=rows[0]['patient_id']
            with self.assertRaisesRegex(ValueError,'leakage'):validate_splits(rows)
    def test_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p=create_demo(d);np.save(Path(d)/'fixture_01.npy',np.load(Path(d)/'fixture_00.npy'))
            with self.assertRaisesRegex(ValueError,'Duplicate'):load_manifest(p)
    def test_bad_shape(self):
        with tempfile.TemporaryDirectory() as d:
            p=create_demo(d);np.save(Path(d)/'fixture_00.npy',np.zeros((12,12)))
            with self.assertRaisesRegex(ValueError,'128x128'):load_manifest(p)
    def test_missing_class(self):
        with tempfile.TemporaryDirectory() as d:
            rows=load_manifest(create_demo(d))
            for r in rows:
                if r['split']=='test':r['label']=0
            with self.assertRaisesRegex(ValueError,'both classes'):validate_splits(rows)
    def test_split_repeatability(self):
        with tempfile.TemporaryDirectory() as d:
            rows=load_manifest(create_demo(d))
            for r in rows:r.pop('split')
            for seed in range(100):
                try:a=assign_splits(rows,seed)
                except ValueError:continue
                self.assertEqual(a,assign_splits(rows,seed));break
            else:self.fail('No valid seed')

class ModelTests(unittest.TestCase):
    def test_hybrid_gradients(self):
        import torch
        from src.hybrid_model import HybridModel
        torch.manual_seed(42);torch.set_num_threads(2)
        m=HybridModel();y=m(torch.rand(2,1,128,128))
        self.assertEqual(tuple(y.shape),(2,1));self.assertTrue(((y>=0)&(y<=1)).all())
        torch.nn.functional.binary_cross_entropy(y,torch.tensor([[0.],[1.]])).backward()
        for name in ['conv1.weight','fc1.weight','quantum.q_layer.weights','fc2.weight']:
            g=dict(m.named_parameters())[name].grad
            self.assertIsNotNone(g);self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum().item(),0)
    def test_checkpoint_roundtrip(self):
        import torch
        from src.hybrid_model import HybridModel
        m=HybridModel().eval();x=torch.rand(1,1,128,128)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.pt';torch.save(m.state_dict(),p);n=HybridModel().eval();n.load_state_dict(torch.load(p,weights_only=True));self.assertTrue(torch.allclose(m(x),n(x)))
    def test_parameter_counts(self):
        from src.hybrid_model import HybridModel
        from src.classical_cnn import ClassicalCNN
        self.assertEqual(sum(p.numel() for p in HybridModel().parameters()),479657)
        self.assertEqual(sum(p.numel() for p in ClassicalCNN().parameters()),7391873)

if __name__=='__main__':unittest.main()
