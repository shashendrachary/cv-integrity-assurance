"""Offline tests for image anomalies and integrity-gated YOLO inference."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from src.anomaly.checks import AnomalyThresholds, analyze_image
from src.integrity.data_checks import sha256_file
from src.model.integrity import register_model
from src.project_context import load_project_context
from src.vision.inference import run_inference


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_path = self.root / "checkerboard.png"
        image = Image.new("L", (64, 48))
        image.putdata(
            [255 if (x // 4 + y // 4) % 2 else 0 for y in range(48) for x in range(64)]
        )
        image.save(self.image_path)
        self.model_path = self.root / "synthetic-model.bin"
        self.model_path.write_bytes(b"synthetic test bytes, not model weights")
        self.registry_path = self.root / "models.json"
        self.thresholds = AnomalyThresholds(
            minimum_width=1,
            minimum_height=1,
            minimum_laplacian_variance=0,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def register_model(self) -> None:
        register_model("test-model", self.model_path, self.registry_path)

    def fake_loader(self, detections: list[object] | None = None) -> Mock:
        model = Mock()
        model.predict.return_value = (
            detections
            if detections is not None
            else [SimpleNamespace(boxes=None, names={})]
        )
        loader = Mock(return_value=model)
        return loader

    def test_anomaly_analysis_reports_properties_for_synthetic_image(self) -> None:
        analysis = analyze_image(self.image_path, self.thresholds)

        self.assertTrue(analysis.valid)
        self.assertEqual((analysis.width, analysis.height), (64, 48))
        self.assertAlmostEqual(analysis.aspect_ratio or 0, 64 / 48)
        self.assertIsNotNone(analysis.mean_brightness)
        self.assertIsNotNone(analysis.laplacian_variance)
        self.assertEqual(analysis.reasons, [])

    def test_missing_and_corrupt_images_are_review_required(self) -> None:
        missing = analyze_image(self.root / "missing.png", self.thresholds)
        corrupt_path = self.root / "corrupt.png"
        corrupt_path.write_bytes(b"not an image")
        corrupt = analyze_image(corrupt_path, self.thresholds)

        self.assertFalse(missing.valid)
        self.assertFalse(corrupt.valid)
        self.assertTrue(missing.reasons)
        self.assertTrue(corrupt.reasons)

    def test_unregistered_model_is_blocked_before_model_loading(self) -> None:
        loader = self.fake_loader()

        result = run_inference(
            self.image_path,
            "unknown-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=loader,
        )

        self.assertEqual(result.status, "REVIEW REQUIRED")
        loader.assert_not_called()

    def test_tampered_model_is_blocked_before_model_loading(self) -> None:
        self.register_model()
        self.model_path.write_bytes(b"tampered synthetic model")
        loader = self.fake_loader()

        result = run_inference(
            self.image_path,
            "test-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=loader,
        )

        self.assertEqual(result.status, "SUSPICIOUS")
        loader.assert_not_called()

    def test_tampered_model_is_blocked_without_affecting_other_registered_model(self) -> None:
        second_model_path = self.root / "second-synthetic-model.bin"
        second_model_path.write_bytes(b"independent synthetic model bytes")
        first_registration = register_model(
            "first-model", self.model_path, self.registry_path
        )
        second_registration = register_model(
            "second-model", second_model_path, self.registry_path
        )
        self.model_path.write_bytes(b"changed first model bytes")
        tampered_loader = self.fake_loader()

        tampered_result = run_inference(
            self.image_path,
            "first-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=tampered_loader,
        )

        self.assertEqual(tampered_result.status, "SUSPICIOUS")
        tampered_loader.assert_not_called()

        independent_loader = self.fake_loader()
        independent_result = run_inference(
            self.image_path,
            "second-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=independent_loader,
        )

        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.assertEqual(independent_result.status, "VERIFIED")
        independent_loader.assert_called_once()
        self.assertEqual(
            independent_result.model_verification.expected_hash,
            second_registration["expected_hash"],
        )
        self.assertEqual(
            independent_result.model_verification.actual_hash,
            second_registration["expected_hash"],
        )
        self.assertEqual(
            registry["models"]["first-model"], first_registration
        )
        self.assertEqual(
            registry["models"]["second-model"], second_registration
        )

    def test_integrity_check_precedes_loading_a_verified_private_snapshot(self) -> None:
        self.register_model()
        loader = self.fake_loader()
        loaded_paths: list[Path] = []
        loader.side_effect = lambda path: loaded_paths.append(path) or loader.return_value

        result = run_inference(
            self.image_path,
            "test-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=loader,
        )

        self.assertEqual(result.status, "VERIFIED")
        loader.assert_called_once()
        snapshot_path = loaded_paths[0]
        self.assertNotEqual(snapshot_path, self.model_path.resolve())
        self.assertTrue(snapshot_path.parent.name.startswith("cv-integrity-model-"))
        self.assertFalse(snapshot_path.exists(), "temporary snapshot should be removed")
        self.assertEqual(result.model_verification.actual_hash, result.model_verification.expected_hash)
        loader.return_value.predict.assert_called_once_with(
            source=str(self.image_path),
            device="cpu",
            conf=0.25,
            verbose=False,
        )

    def test_inference_uses_selected_project_root_for_relative_model_path(self) -> None:
        project_root = self.root / "alternative-project"
        model_path = project_root / "models" / "relative-model.bin"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"relative project model bytes")
        registry_path = project_root / "models" / "model_registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": {
                        "relative-model": {
                            "model_id": "relative-model",
                            "file_path": "models/relative-model.bin",
                            "expected_hash": hashlib.sha256(
                                model_path.read_bytes()
                            ).hexdigest(),
                            "registered_at": "2026-01-01T00:00:00+00:00",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        configuration_path = project_root / "project.yaml"
        configuration_path.write_text(
            "project:\n"
            "  organization_id: test-org\n"
            "  project_id: alternative\n"
            "paths:\n"
            "  images: images\n"
            "  labels: labels\n"
            "  model_registry: models/model_registry.json\n"
            "  reports: reports\n",
            encoding="utf-8",
        )
        context = load_project_context(configuration_path)
        loader = self.fake_loader()

        result = run_inference(
            self.image_path,
            "relative-model",
            context.model_registry,
            anomaly_thresholds=self.thresholds,
            model_loader=loader,
            project_root=context.base_path,
        )

        self.assertEqual(result.status, "VERIFIED")
        self.assertEqual(result.model_verification.file_path, str(model_path.resolve()))
        loader.assert_called_once()

    def test_changed_snapshot_is_rejected_before_loader(self) -> None:
        self.register_model()
        loader = self.fake_loader()
        original_copy = shutil.copyfile

        def copy_then_tamper(source: Path, destination: Path) -> Path:
            copied = original_copy(source, destination)
            with Path(destination).open("ab") as file_handle:
                file_handle.write(b"tampered during snapshot")
            return copied

        with patch("src.vision.inference.shutil.copyfile", side_effect=copy_then_tamper):
            result = run_inference(
                self.image_path,
                "test-model",
                self.registry_path,
                anomaly_thresholds=self.thresholds,
                model_loader=loader,
            )

        self.assertEqual(result.status, "SUSPICIOUS")
        self.assertNotEqual(
            result.model_verification.expected_hash,
            result.model_verification.actual_hash,
        )
        loader.assert_not_called()

    def test_snapshot_change_before_loader_fails_final_hash_check(self) -> None:
        self.register_model()
        loader = self.fake_loader()
        original_hash = sha256_file
        hash_calls = 0

        def hash_with_preload_change(path: Path) -> str:
            nonlocal hash_calls
            hash_calls += 1
            if hash_calls == 2:
                with path.open("ab") as file_handle:
                    file_handle.write(b"changed immediately before load")
            return original_hash(path)

        with patch(
            "src.vision.inference.sha256_file",
            side_effect=hash_with_preload_change,
        ):
            result = run_inference(
                self.image_path,
                "test-model",
                self.registry_path,
                anomaly_thresholds=self.thresholds,
                model_loader=loader,
            )

        self.assertEqual(result.status, "SUSPICIOUS")
        self.assertIn("changed before loading", result.model_verification.reason)
        loader.assert_not_called()

    def test_inference_exception_requires_review(self) -> None:
        self.register_model()
        loader = Mock(side_effect=RuntimeError("model loader failed"))

        result = run_inference(
            self.image_path,
            "test-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=loader,
        )

        self.assertEqual(result.status, "REVIEW REQUIRED")
        self.assertFalse(result.inference_succeeded)
        self.assertIn("model loader failed", result.reasons[0])

    def test_empty_detection_result_is_reported_without_error(self) -> None:
        self.register_model()
        loader = self.fake_loader()

        result = run_inference(
            self.image_path,
            "test-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=loader,
        )

        self.assertEqual(result.status, "VERIFIED")
        self.assertEqual(result.detections, [])
        self.assertIn("No objects were detected.", result.reasons)

    def test_invalid_image_prevents_model_verification_and_loading(self) -> None:
        loader = self.fake_loader()
        result = run_inference(
            self.root / "missing-image.png",
            "unknown-model",
            self.registry_path,
            model_loader=loader,
        )

        self.assertEqual(result.status, "REVIEW REQUIRED")
        loader.assert_not_called()

    def test_detection_fields_are_returned(self) -> None:
        self.register_model()

        class FakeBoxes:
            cls = [SimpleNamespace(item=lambda: 2)]
            conf = [SimpleNamespace(item=lambda: 0.91)]
            xyxy = [SimpleNamespace(tolist=lambda: [1, 2, 30, 40])]

            def __len__(self) -> int:
                return 1

        box = FakeBoxes()
        prediction = SimpleNamespace(boxes=box, names={2: "car"})
        loader = self.fake_loader([prediction])

        result = run_inference(
            self.image_path,
            "test-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=loader,
        )

        self.assertEqual(len(result.detections), 1, result.reasons)
        self.assertEqual(result.detections[0].class_name, "car")
        self.assertEqual(result.detections[0].class_id, 2)
        self.assertAlmostEqual(result.detections[0].confidence, 0.91)
        self.assertEqual(result.detections[0].box_xyxy, (1.0, 2.0, 30.0, 40.0))


if __name__ == "__main__":
    unittest.main()
