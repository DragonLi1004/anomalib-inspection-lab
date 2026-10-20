"""Regression tests for memory-efficient upstream execution and reversible setup."""
import unittest
from baseline import ROOT
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from exact_search import configure_exact

class ExactExecutionTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(71)
        model=PatchcoreModel(backbone='resnet18.tv_in1k',layers=['layer2','layer3'],pre_trained=False,num_neighbors=9).eval()
        model.memory_bank=torch.rand(41,384)
        return model

    def test_lifetime_mode_preserves_raw_outputs_and_checkpoint_tensors(self):
        m=self.make_model();x=torch.rand(1,3,64,64)
        before={k:v.clone() for k,v in m.state_dict().items()}
        with torch.inference_mode():
            original=m(x);configure_exact(m); optimized=m(x)
        self.assertTrue(torch.equal(original.pred_score,optimized.pred_score))
        self.assertTrue(torch.equal(original.anomaly_map,optimized.anomaly_map))
        self.assertEqual(set(before),set(m.state_dict()))
        for key,value in m.state_dict().items():self.assertTrue(torch.equal(value,before[key]),key)
        configure_exact(m,enabled=False)
        with torch.inference_mode():restored=m(x)
        self.assertTrue(torch.equal(original.anomaly_map,restored.anomaly_map))

    def test_chunked_neighbors_preserve_ties_and_topk(self):
        m=self.make_model();m.memory_bank=torch.tensor([[0.,0.],[1.,0.],[1.,0.],[3.,0.]])
        queries=torch.tensor([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[4.,0.]])
        for k in [1,3]:
            configure_exact(m,enabled=False); expected=m.nearest_neighbors(queries,k)
            configure_exact(m,query_chunk=2); actual=m.nearest_neighbors(queries,k)
            for a,b in zip(actual,expected):self.assertTrue(torch.equal(a,b))

    def test_training_delegates_to_upstream_and_invalid_chunk_rejected(self):
        m=self.make_model().train();x=torch.rand(1,3,64,64)
        with torch.inference_mode():
            before=m(x);m.embedding_store.clear();configure_exact(m);after=m(x)
        self.assertTrue(torch.equal(before,after));self.assertEqual(len(m.embedding_store),1)
        for chunk in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):configure_exact(m,chunk)

if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
