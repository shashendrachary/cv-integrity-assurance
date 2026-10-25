"""Offline end-to-end workflow and PDF report tests."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

from PIL import Image

from src.anomaly.checks import AnomalyThresholds
from src.database.postgres import DatabaseOperationError
from src.model.integrity import register_model
from src.provenance.workflow import classify_assessment, run_assessment
from src.reports.pdf_report import generate_pdf_report
from src.vision.inference import run_inference


class AssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.image_path = self.images / "scene.png"
        image = Image.new("RGB", (64, 48), (80, 120, 160))
        image.save(self.image_path)
        self.model_path = self.root / "test-model.bin"
        self.model_path.write_bytes(b"synthetic test bytes, not model weights")
        self.registry_path = self.root / "registry.json"
        self.report_directory = self.root / "reports"
        self.thresholds = AnomalyThresholds(
            minimum_width=1,
            minimum_height=1,
            minimum_laplacian_variance=0,
        )
        self.database_settings = {
            "host": "localhost",
            "port": 5433,
            "dbname": "offline_test",
            "user": "test_user",
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def create_model_loader(self) -> Mock:
        class EmptyBoxes:
            def __len__(self) -> int:
                return 0

        model = Mock()
        model.predict.return_value = [
            SimpleNamespace(boxes=EmptyBoxes(), names={})
        ]
        return Mock(return_value=model)

    def assess(self, *, settings=None, inserter=None, image_path=None, model_id="test-model"):
        return run_assessment(
            image_path or self.image_path,
            model_id,
            self.registry_path,
            self.report_directory,
            confidence=0.25,
            anomaly_thresholds=self.thresholds,
            database_settings=settings,
            model_loader=self.create_model_loader(),
            record_inserter=inserter or Mock(),
        )

    def test_end_to_end_generates_pdf_and_persists_record(self) -> None:
        register_model("test-model", self.model_path, self.registry_path)
        inserter = Mock()

        assessment = self.assess(settings=self.database_settings, inserter=inserter)

        self.assertEqual(assessment.overall_status, "VERIFIED")
        self.assertEqual(assessment.persistence_status, "saved to PostgreSQL")
        self.assertTrue(assessment.inference.inference_succeeded)
        self.assertTrue(assessment.report_path.is_file())
        self.assertTrue(assessment.report_path.read_bytes().startswith(b"%PDF-"))
        self.assertEqual(assessment.provenance.run_id, assessment.report_path.stem.removeprefix("assurance-"))
        inserter.assert_called_once()

    def test_database_unavailable_still_generates_pdf_and_requires_review(self) -> None:
        register_model("test-model", self.model_path, self.registry_path)

        assessment = self.assess(settings=None)

        self.assertTrue(assessment.inference.inference_succeeded)
        self.assertEqual(assessment.inference.status, "VERIFIED")
        self.assertEqual(assessment.overall_status, "REVIEW REQUIRED")
        self.assertIn("skipped", assessment.persistence_status)
        self.assertTrue(assessment.report_path.is_file())

    def test_database_failure_keeps_inference_result_and_generates_pdf(self) -> None:
        register_model("test-model", self.model_path, self.registry_path)

        def fail_insert(*_args, **_kwargs):
            raise DatabaseOperationError("PostgreSQL unavailable")

        assessment = self.assess(
            settings=self.database_settings,
            inserter=fail_insert,
        )

        self.assertTrue(assessment.inference.inference_succeeded)
        self.assertEqual(assessment.overall_status, "REVIEW REQUIRED")
        self.assertIn("not saved", assessment.persistence_status)
        self.assertTrue(assessment.report_path.is_file())

    def test_tampered_model_stays_suspicious_and_loader_is_blocked(self) -> None:
        register_model("test-model", self.model_path, self.registry_path)
        self.model_path.write_bytes(b"tampered")
        loader = self.create_model_loader()
        assessment = run_assessment(
            self.image_path,
            "test-model",
            self.registry_path,
            self.report_directory,
            confidence=0.25,
            anomaly_thresholds=self.thresholds,
            database_settings=None,
            model_loader=loader,
        )

        self.assertEqual(assessment.overall_status, "SUSPICIOUS")
        loader.assert_not_called()
        self.assertTrue(assessment.report_path.is_file())

    def test_missing_image_and_unregistered_model_get_reports(self) -> None:
        missing_image = self.root / "missing.png"

        missing_image_assessment = self.assess(image_path=missing_image)
        missing_model_assessment = self.assess(model_id="unregistered")

        self.assertEqual(missing_image_assessment.overall_status, "REVIEW REQUIRED")
        self.assertTrue(missing_image_assessment.report_path.is_file())
        self.assertEqual(missing_model_assessment.overall_status, "REVIEW REQUIRED")
        self.assertTrue(missing_model_assessment.report_path.is_file())

    def test_missing_registered_model_file_gets_review_report(self) -> None:
        register_model("test-model", self.model_path, self.registry_path)
        self.model_path.unlink()
        loader = self.create_model_loader()

        assessment = run_assessment(
            self.image_path,
            "test-model",
            self.registry_path,
            self.report_directory,
            confidence=0.25,
            anomaly_thresholds=self.thresholds,
            database_settings=None,
            model_loader=loader,
        )

        self.assertEqual(assessment.overall_status, "REVIEW REQUIRED")
        loader.assert_not_called()
        self.assertTrue(assessment.report_path.is_file())

    def test_pdf_path_is_unique_for_same_run_identifier(self) -> None:
        register_model("test-model", self.model_path, self.registry_path)
        inference = run_inference(
            self.image_path,
            "test-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=self.create_model_loader(),
        )
        from src.provenance.records import create_provenance_record

        record = create_provenance_record(self.image_path, inference, {"confidence": 0.25})

        first = generate_pdf_report(
            record, "VERIFIED", "saved to PostgreSQL", self.report_directory
        )
        second = generate_pdf_report(
            record, "VERIFIED", "saved to PostgreSQL", self.report_directory
        )

        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())
        self.assertIn(record.run_id.encode("ascii"), first.read_bytes())

    def test_pdf_generation_failure_does_not_change_existing_report(self) -> None:
        register_model("test-model", self.model_path, self.registry_path)
        inference = run_inference(
            self.image_path,
            "test-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=self.create_model_loader(),
        )
        from src.provenance.records import create_provenance_record

        record = create_provenance_record(self.image_path, inference, {"confidence": 0.25})
        existing = generate_pdf_report(
            record, "VERIFIED", "saved", self.report_directory
        )
        original_bytes = existing.read_bytes()

        with patch(
            "reportlab.platypus.SimpleDocTemplate.build",
            side_effect=OSError("simulated PDF failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not generate PDF"):
                generate_pdf_report(
                    record, "VERIFIED", "saved", self.report_directory
                )

        self.assertEqual(existing.read_bytes(), original_bytes)
        self.assertEqual(list(self.report_directory.glob("*.pdf")), [existing])

    def test_classification_prioritizes_tampering_over_other_review_causes(self) -> None:
        register_model("test-model", self.model_path, self.registry_path)
        self.model_path.write_bytes(b"changed bytes")
        inference = run_inference(
            self.image_path,
            "test-model",
            self.registry_path,
            anomaly_thresholds=self.thresholds,
            model_loader=self.create_model_loader(),
        )

        status, _ = classify_assessment(inference, persistence_succeeded=False)

        self.assertEqual(status, "SUSPICIOUS")


if __name__ == "__main__":
    unittest.main()
