"""Focused tests for recursive batch image assessment."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from assurance import (
    BatchImageDiscoveryError,
    build_batch_report_data,
    build_parser,
    discover_batch_images,
    main,
    run_batch_assessments,
)
from src.anomaly.checks import AnomalyThresholds
from src.project_context import load_project_context
from src.reports.pdf_report import (
    BatchImageReportItem,
    BatchReportSummary,
    generate_batch_pdf_report,
)


class BatchImageDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_discovers_supported_images_recursively_in_sorted_order(self) -> None:
        (self.root / "nested").mkdir()
        paths = [
            self.root / "z.JPG",
            self.root / "nested" / "b.png",
            self.root / "a.webp",
            self.root / "ignored.txt",
        ]
        for path in paths:
            path.write_bytes(b"test")

        self.assertEqual(
            discover_batch_images(self.root),
            [self.root / "a.webp", self.root / "nested" / "b.png", self.root / "z.JPG"],
        )

    def test_missing_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(BatchImageDiscoveryError, "does not exist"):
            discover_batch_images(self.root / "missing")

    def test_non_directory_path_is_rejected(self) -> None:
        file_path = self.root / "not-a-directory.txt"
        file_path.write_text("test", encoding="utf-8")
        with self.assertRaisesRegex(BatchImageDiscoveryError, "not a directory"):
            discover_batch_images(file_path)

    def test_directory_without_supported_images_is_rejected(self) -> None:
        (self.root / "notes.txt").write_text("no images", encoding="utf-8")
        with self.assertRaisesRegex(BatchImageDiscoveryError, "No supported images"):
            discover_batch_images(self.root)

    def test_unreadable_subdirectory_is_reported(self) -> None:
        (self.root / "image.jpg").write_bytes(b"test")

        def walk_with_error(_path, *, onerror):
            onerror(PermissionError("access denied"))
            return []

        with patch("assurance.os.walk", side_effect=walk_with_error):
            with self.assertRaisesRegex(BatchImageDiscoveryError, "Unable to read"):
                discover_batch_images(self.root)


class BatchAssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.context = load_project_context(self._write_project_config())

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_project_config(self) -> Path:
        config = self.root / "project.yaml"
        config.write_text(
            "project:\n"
            "  organization_id: org-batch\n"
            "  project_id: project-batch\n"
            "paths:\n"
            "  images: images\n"
            "  labels: labels\n"
            "  model_registry: registry.json\n"
            "  reports: reports\n",
            encoding="utf-8",
        )
        return config

    def test_continues_after_image_failure_and_counts_unverified_results(self) -> None:
        image_paths = [self.root / "a.png", self.root / "b.png", self.root / "c.png"]
        runner = Mock(
            side_effect=[
                RuntimeError("image decode failed"),
                SimpleNamespace(overall_status="REVIEW REQUIRED", report_path=None),
                SimpleNamespace(
                    overall_status="VERIFIED",
                    report_path=self.root / "reports" / "c.pdf",
                ),
            ]
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            entries = run_batch_assessments(
                image_paths,
                model_id="selected-model",
                registry_path=self.root / "registry.json",
                report_directory=self.root / "reports",
                confidence=0.4,
                anomaly_thresholds=AnomalyThresholds(),
                database_settings=None,
                database_note=None,
                project_context=self.context,
                assessment_runner=runner,
            )

        self.assertEqual([entry.status for entry in entries], [
            "FAILED",
            "REVIEW REQUIRED",
            "VERIFIED",
        ])
        self.assertEqual(runner.call_count, 3)
        self.assertTrue(
            all(call.args[1] == "selected-model" for call in runner.call_args_list)
        )
        self.assertTrue(
            all(call.kwargs["generate_report"] is False
                for call in runner.call_args_list)
        )
        self.assertIn("Status: FAILED", output.getvalue())
        self.assertIn("Status: REVIEW REQUIRED", output.getvalue())
        self.assertNotIn("PDF report:", output.getvalue())

    def test_batch_creates_one_consolidated_pdf_and_sidecar_for_multiple_images(self) -> None:
        image_paths = [self.root / "one.png", self.root / "two.png"]
        for path in image_paths:
            Image.new("RGB", (20, 12), "white").save(path)
        inference = SimpleNamespace(
            image_analysis=SimpleNamespace(
                valid=True,
                width=20,
                height=12,
                reasons=[],
            ),
            model_verification=SimpleNamespace(status="VERIFIED"),
            inference_succeeded=True,
            reasons=[],
            detections=[],
        )

        def result_for(path, detection):
            return SimpleNamespace(
                overall_status="VERIFIED",
                report_path=None,
                inference=SimpleNamespace(
                    **{
                        **inference.__dict__,
                        "detections": [],
                    }
                ),
                provenance=SimpleNamespace(
                    image_sha256=f"hash-{path.name}",
                    image_metrics={"width": 20, "height": 12},
                    reasons=[],
                    detections=[detection],
                ),
            )

        runner = Mock(
            side_effect=[
                result_for(
                    image_paths[0],
                    {"class_name": "car", "class_id": 1, "confidence": 0.9},
                ),
                result_for(
                    image_paths[1],
                    {"class_name": "person", "class_id": 0, "confidence": 0.8},
                ),
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            entries = run_batch_assessments(
                image_paths,
                model_id="selected-model",
                registry_path=self.root / "registry.json",
                report_directory=self.root / "reports",
                confidence=0.4,
                anomaly_thresholds=AnomalyThresholds(),
                database_settings=None,
                database_note=None,
                project_context=self.context,
                assessment_runner=runner,
            )

        items, summary = build_batch_report_data(
            image_paths, entries, "selected-model"
        )
        pdf_path, sidecar_path = generate_batch_pdf_report(
            items, summary, self.root / "reports"
        )
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        self.assertEqual(len(list((self.root / "reports").glob("*.pdf"))), 1)
        self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF-"))
        self.assertEqual([item["filename"] for item in sidecar["images"]], [
            "one.png",
            "two.png",
        ])
        self.assertEqual(sidecar["ai_results"]["detection_count"], 2)
        self.assertEqual(
            sidecar["ai_results"]["object_class_counts"],
            {"car": 1, "person": 1},
        )
        self.assertEqual(
            [call.kwargs["generate_report"] for call in runner.call_args_list],
            [False, False],
        )

    def test_mixed_outcome_counts_include_review_failure_and_inference_failure(self) -> None:
        image_paths = [self.root / "verified.png", self.root / "review.png", self.root / "broken.png"]
        for path in image_paths[:2]:
            Image.new("RGB", (32, 24), "white").save(path)
        detections = [
            {"class_name": "car", "class_id": 1},
            {"class_name": "car", "class_id": 1},
            {"class_name": "person", "class_id": 0},
        ]

        def make_assessment(path, status, inference_succeeded, detection_list):
            return SimpleNamespace(
                overall_status=status,
                inference=SimpleNamespace(
                    image_analysis=SimpleNamespace(
                        valid=True,
                        width=32,
                        height=24,
                        reasons=[] if status == "VERIFIED" else ["quality warning"],
                    ),
                    model_verification=SimpleNamespace(status="VERIFIED"),
                    inference_succeeded=inference_succeeded,
                    reasons=[],
                ),
                provenance=SimpleNamespace(
                    image_sha256="same-hash",
                    image_metrics={"width": 32, "height": 24},
                    reasons=[] if status == "VERIFIED" else ["quality warning"],
                    detections=detection_list,
                ),
            )

        runner = Mock(
            side_effect=[
                make_assessment(image_paths[0], "VERIFIED", True, detections),
                make_assessment(image_paths[1], "REVIEW REQUIRED", False, []),
                RuntimeError("unreadable image"),
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            entries = run_batch_assessments(
                image_paths,
                model_id="selected-model",
                registry_path=self.root / "registry.json",
                report_directory=self.root / "reports",
                confidence=0.4,
                anomaly_thresholds=AnomalyThresholds(),
                database_settings=None,
                database_note=None,
                project_context=self.context,
                assessment_runner=runner,
            )

        items, summary = build_batch_report_data(
            image_paths, entries, "selected-model"
        )
        pdf_path, sidecar_path = generate_batch_pdf_report(
            items, summary, self.root / "reports"
        )
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        self.assertEqual(
            (
                summary.discovered,
                summary.processed,
                summary.successful,
                summary.review_required,
                summary.failures,
                summary.inference_failures,
            ),
            (3, 3, 1, 1, 1, 2),
        )
        self.assertEqual(sidecar["ai_results"]["detection_count"], 3)
        self.assertEqual(
            sidecar["ai_results"]["object_class_counts"],
            {"car": 2, "person": 1},
        )
        self.assertTrue(pdf_path.is_file())
        self.assertEqual(len(list((self.root / "reports").glob("*.pdf"))), 1)

    def test_empty_batch_is_rejected_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "No images were discovered"):
            build_batch_report_data([], [], "selected-model")
        with self.assertRaisesRegex(ValueError, "No images were supplied"):
            run_batch_assessments(
                [],
                model_id="selected-model",
                registry_path=self.root / "registry.json",
                report_directory=self.root / "reports",
                confidence=0.4,
                anomaly_thresholds=AnomalyThresholds(),
                database_settings=None,
                database_note=None,
                project_context=self.context,
            )

    def test_batch_pdf_includes_n_a_for_unavailable_properties(self) -> None:
        item = BatchImageReportItem(
            filename="broken.png",
            image_sha256=None,
            width=None,
            height=None,
            image_format=None,
            duplicate_status="N/A",
            quality_flags=("unreadable",),
            status="FAILED",
            model_verification_status="N/A",
            inference_succeeded=False,
            reasons=("unreadable image",),
            detections=(),
        )
        pdf_path, _ = generate_batch_pdf_report(
            [item],
            BatchReportSummary(1, 1, 0, 0, 1, 1, "selected-model", "N/A"),
            self.root / "reports",
        )
        self.assertTrue(pdf_path.is_file())
        self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF-"))

    def test_cli_accepts_options_and_propagates_model_id_and_overrides(self) -> None:
        images = self.root / "images"
        images.mkdir()
        image_path = images / "sample.png"
        image_path.write_bytes(b"test")
        config_path = self.context.configuration_path
        thresholds = AnomalyThresholds()
        assessment = SimpleNamespace(
            overall_status="VERIFIED",
            report_path=self.root / "reports" / "sample.pdf",
        )
        logger = Mock()

        parsed = build_parser().parse_args(
            [
                "assess-batch",
                "--id",
                "selected-model",
                "--images-dir",
                "images",
                "--conf",
                "0.63",
                "--registry",
                "alternate-registry.json",
                "--project-config",
                str(config_path),
            ]
        )
        self.assertEqual(parsed.id, "selected-model")
        self.assertEqual(parsed.images_dir, Path("images"))
        self.assertEqual(parsed.conf, 0.63)
        self.assertEqual(parsed.registry, Path("alternate-registry.json"))
        self.assertEqual(parsed.project_config, config_path)

        output = io.StringIO()
        with (
            patch("assurance.configure_logging", return_value=logger),
            patch(
                "assurance.load_inference_settings",
                return_value=(thresholds, 0.25),
            ),
            patch("assurance.load_database_settings", return_value=None),
            patch("assurance.run_assessment", return_value=assessment) as runner,
            patch(
                "assurance.build_batch_report_data",
                return_value=(
                    [],
                    BatchReportSummary(1, 1, 1, 0, 0, 0, "selected-model", "VERIFIED"),
                ),
            ),
            patch(
                "assurance.generate_batch_pdf_report",
                return_value=(
                    self.root / "reports" / "batch.pdf",
                    self.root / "reports" / "batch.json",
                ),
            ),
            contextlib.redirect_stdout(output),
        ):
            exit_code = main(
                [
                    "assess-batch",
                    "--id",
                    "selected-model",
                    "--images-dir",
                    "images",
                    "--conf",
                    "0.63",
                    "--registry",
                    "alternate-registry.json",
                    "--project-config",
                    str(config_path),
                ]
            )

        self.assertEqual(exit_code, 0)
        runner.assert_called_once()
        self.assertEqual(runner.call_args.args[:3], (
            image_path,
            "selected-model",
            self.root / "alternate-registry.json",
        ))
        self.assertEqual(runner.call_args.kwargs["confidence"], 0.63)
        self.assertEqual(runner.call_args.kwargs["project_context"], self.context)
        self.assertFalse(runner.call_args.kwargs["generate_report"])
        self.assertIn(
            "discovered=1, processed=1, successful=1, review-required=0, failures=0",
            output.getvalue(),
        )
        self.assertIn("Consolidated PDF report:", output.getvalue())

    def test_cli_returns_nonzero_and_reports_failed_totals(self) -> None:
        self.context.images_directory.mkdir(parents=True)
        (self.context.images_directory / "needs-review.png").write_bytes(b"test")
        logger = Mock()
        output = io.StringIO()

        with (
            patch("assurance.configure_logging", return_value=logger),
            patch(
                "assurance.load_inference_settings",
                return_value=(AnomalyThresholds(), 0.25),
            ),
            patch("assurance.load_database_settings", return_value=None),
            patch(
                "assurance.run_assessment",
                return_value=SimpleNamespace(
                    overall_status="REVIEW REQUIRED",
                    report_path=None,
                ),
            ),
            patch(
                "assurance.build_batch_report_data",
                return_value=(
                    [],
                    BatchReportSummary(1, 1, 0, 1, 0, 0, "selected-model", "VERIFIED"),
                ),
            ),
            patch(
                "assurance.generate_batch_pdf_report",
                return_value=(
                    self.root / "reports" / "batch.pdf",
                    self.root / "reports" / "batch.json",
                ),
            ),
            contextlib.redirect_stdout(output),
        ):
            exit_code = main(
                [
                    "assess-batch",
                    "--id",
                    "selected-model",
                    "--images-dir",
                    "images",
                    "--project-config",
                    str(self.context.configuration_path),
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn(
            "discovered=1, processed=1, successful=0, review-required=1, failures=0",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
