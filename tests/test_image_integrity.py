"""Focused tests for trusted input-image SHA-256 verification."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from src.anomaly.checks import AnomalyThresholds
from src.integrity.image_baselines import register_image, verify_image
from src.model.integrity import register_model
from src.vision.inference import run_inference


class ImageIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_path = self.root / "trusted.png"
        Image.new("RGB", (32, 32), (20, 100, 180)).save(self.image_path)
        self.image_registry = self.root / "image_registry.json"
        self.model_path = self.root / "model.bin"
        self.model_path.write_bytes(b"synthetic model")
        self.model_registry = self.root / "model_registry.json"
        register_model("test-model", self.model_path, self.model_registry)
        self.thresholds = AnomalyThresholds(
            minimum_width=1,
            minimum_height=1,
            minimum_laplacian_variance=0,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def loader(self) -> Mock:
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=None, names={})]
        return Mock(return_value=model)

    def test_trusted_unchanged_image_is_verified(self) -> None:
        register_image(self.image_path, self.image_registry)

        result = run_inference(
            self.image_path,
            "test-model",
            self.model_registry,
            anomaly_thresholds=self.thresholds,
            model_loader=self.loader(),
            image_registry_path=self.image_registry,
        )

        self.assertEqual(result.image_verification.status, "VERIFIED")
        self.assertEqual(result.status, "VERIFIED")
        self.assertTrue(result.inference_succeeded)

    def test_modified_image_is_suspicious_without_blocking_detection(self) -> None:
        register_image(self.image_path, self.image_registry)
        Image.new("RGB", (32, 32), (200, 40, 10)).save(self.image_path)
        loader = self.loader()

        result = run_inference(
            self.image_path,
            "test-model",
            self.model_registry,
            anomaly_thresholds=self.thresholds,
            model_loader=loader,
            image_registry_path=self.image_registry,
        )

        self.assertEqual(result.image_verification.status, "SUSPICIOUS")
        self.assertEqual(result.status, "SUSPICIOUS")
        self.assertTrue(result.inference_succeeded)
        loader.assert_called_once()

    def test_missing_baseline_requires_review(self) -> None:
        verification = verify_image(self.image_path, self.image_registry)
        loader = self.loader()

        result = run_inference(
            self.image_path,
            "test-model",
            self.model_registry,
            anomaly_thresholds=self.thresholds,
            model_loader=loader,
            image_registry_path=self.image_registry,
        )

        self.assertEqual(verification.status, "REVIEW REQUIRED")
        self.assertEqual(result.image_verification.status, "REVIEW REQUIRED")
        self.assertEqual(result.status, "REVIEW REQUIRED")
        self.assertTrue(result.inference_succeeded)
        loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
