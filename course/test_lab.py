"""Focused integrity checks for projection and the local HTTP boundary."""
import json
import threading
import unittest
import urllib.error
import urllib.request

from baseline import ROOT
import torch
from compact_patchcore import CompactPatchcore, fit_projection
import gui


class ProjectionTests(unittest.TestCase):
    def test_full_rank_preserves_pairwise_distances(self):
        generator = torch.Generator().manual_seed(17)
        bank = torch.randn(64, 16, generator=generator)
        queries = torch.randn(12, 16, generator=generator)
        mean, basis, retained = fit_projection(bank, 16)
        torch.testing.assert_close(torch.cdist(bank, queries), torch.cdist((bank-mean)@basis, (queries-mean)@basis), atol=2e-5, rtol=1e-5)
        self.assertAlmostEqual(float(retained), 1.0, places=6)

    def test_reduced_projection_orthonormal_and_training_mean(self):
        bank = torch.arange(512, dtype=torch.float32).reshape(32, 16)
        mean, basis, _ = fit_projection(bank, 8)
        torch.testing.assert_close(mean, bank.mean(0))
        torch.testing.assert_close(basis.T @ basis, torch.eye(8), atol=1e-6, rtol=1e-5)
        with self.assertRaises(ValueError):
            fit_projection(bank, 17)

    def test_model_construction_preserves_upstream_rng_and_weights(self):
        from anomalib.models import Patchcore
        kwargs = dict(backbone="resnet18.tv_in1k", layers=["layer2", "layer3"], pre_trained=False,
                      pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
        torch.manual_seed(42)
        original = Patchcore(**kwargs)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        compact = CompactPatchcore(**kwargs)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for name, tensor in original.model.state_dict().items():
            torch.testing.assert_close(tensor, compact.model.state_dict()[name])

    def test_checkpoint_preserves_query_projection(self):
        paths = sorted((ROOT / "course/outputs").glob("*-pca96-*/baseline.ckpt"))
        if not paths:
            self.skipTest("Run a PCA experiment first")
        model = CompactPatchcore.load_from_checkpoint(paths[-1], map_location="cpu", weights_only=False)
        state = torch.load(paths[-1], map_location="cpu", weights_only=False)["state_dict"]
        torch.testing.assert_close(model.model.projection_basis, state["model.projection_basis"])
        torch.testing.assert_close(model.model.memory_bank, state["model.memory_bank"])
        features = {"layer2": torch.rand(1,128,7,7), "layer3": torch.rand(1,256,7,7)}
        raw = torch.cat([features["layer2"], features["layer3"]], dim=1)
        expected = (raw.permute(0,2,3,1)-model.model.projection_mean) @ model.model.projection_basis
        actual = model.model.generate_embedding(features).permute(0,2,3,1)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)


class LocalServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = gui.ThreadingHTTPServer(("127.0.0.1", 0), gui.Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_state_and_csv_available(self):
        with urllib.request.urlopen(self.url + "/api/state") as response:
            self.assertIn("runs", json.load(response))
        with urllib.request.urlopen(self.url + "/api/export") as response:
            self.assertIn(b"image_AUROC", response.read())

    def test_mutation_requires_local_token(self):
        request = urllib.request.Request(self.url + "/api/job", data=b'{}', headers={"Content-Type":"application/json"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_files_cannot_escape_output_directory(self):
        for path in ("/api/file?path=.runtime/gh/hosts.yml", "/api/file?path=course/outputs/../../pyproject.toml", "/../gui.py"):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(self.url + path)
            self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
