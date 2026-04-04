import torch
import unittest

from backend.models.hybrid_qcnn import HybridQCNN
from backend.training.train_v2 import analyze_class_collapse


class TestBehavior(unittest.TestCase):
    def test_hybrid_qcnn_forward_shape_cpu_safe(self):
        """VRAM-safe behavior gate: tiny CPU-only forward pass keeps CI lightweight."""
        torch.manual_seed(42)
        device = torch.device("cpu")

        model = HybridQCNN(
            input_dim=384,
            n_qubits=4,
            n_layers=1,
            n_classes=3,
            use_qcnn=True,
            dropout=0.0,
        ).to(device)
        model.eval()

        dummy_embeddings = torch.randn(2, 384, device=device)
        lang_ids = torch.zeros(2, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(dummy_embeddings, lang_ids=lang_ids)

        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_collapse_analysis_flags_zero_recall_classes(self):
        """Detect mode-collapse when predictions are stuck in one class."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [1, 1, 1, 1, 1, 1]

        report = analyze_class_collapse(y_true, y_pred, num_classes=3)

        self.assertTrue(report["collapse_detected"])
        self.assertIn("negative", report["collapsed_classes"])
        self.assertIn("positive", report["collapsed_classes"])
        self.assertEqual(report["per_class_recall"]["neutral"], 1.0)


if __name__ == "__main__":
    unittest.main()
