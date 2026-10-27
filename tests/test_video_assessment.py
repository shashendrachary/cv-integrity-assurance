"""Focused tests for sequential video-frame assessment."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from assurance import (
    VideoAssessmentError,
    VideoAssessmentOutcome,
    build_parser,
    main,
    run_video_assessments,
    validate_video_interval,
)
from src.anomaly.checks import AnomalyThresholds
from src.project_context import load_project_context


class VideoAssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.video_path = self.root / "source.mp4"
        self.video_path.write_bytes(b"synthetic test video")
        self.config_path = self.root / "project.yaml"
        self.config_path.write_text(
            "project:\n"
            "  organization_id: org-video\n"
            "  project_id: video-project\n"
            "paths:\n"
            "  images: images\n"
            "  labels: labels\n"
            "  model_registry: registry.json\n"
            "  reports: reports\n",
            encoding="utf-8",
        )
        self.context = load_project_context(self.config_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_capture(self, *, frame_rate=2.0, frames=0, opened=True):
        capture = Mock()
        capture.isOpened.return_value = opened
        capture.get.side_effect = lambda prop: (
            frame_rate if prop == cv2.CAP_PROP_FPS else 0
        )
        capture.read.side_effect = [
            *((True, object()) for _ in range(frames)),
            (False, None),
        ]
        return capture

    def run_video(self, capture, assessment_runner, frame_writer=None, interval=1.0):
        if frame_writer is None:
            def frame_writer(path, _frame):
                Path(path).write_bytes(b"temporary frame")
                return True

        return run_video_assessments(
            self.video_path,
            interval_seconds=interval,
            model_id="video-model",
            registry_path=self.root / "registry.json",
            report_directory=self.root / "reports",
            confidence=0.4,
            anomaly_thresholds=AnomalyThresholds(),
            project_context=self.context,
            assessment_runner=assessment_runner,
            capture_factory=lambda _path: capture,
            frame_writer=frame_writer,
        )

    def make_assessment(
        self,
        status="VERIFIED",
        *,
        detections=(),
        findings=(),
        inference_succeeded=True,
    ):
        detection_list = list(detections)
        return SimpleNamespace(
            overall_status=status,
            report_path=None,
            inference=SimpleNamespace(
                detections=[object() for _ in detection_list],
                reasons=list(findings),
                inference_succeeded=inference_succeeded,
                model_verification=SimpleNamespace(
                    status="VERIFIED",
                    model_id="video-model",
                ),
                image_analysis=SimpleNamespace(
                    reasons=list(findings),
                ),
            ),
            provenance=SimpleNamespace(
                detections=detection_list,
                reasons=list(findings),
            ),
        )

    def test_interval_must_be_finite_and_greater_than_zero(self) -> None:
        for interval in (0, -0.5, float("inf"), float("nan")):
            with self.subTest(interval=interval):
                with self.assertRaises(VideoAssessmentError):
                    validate_video_interval(interval)

    def test_samples_deterministic_frame_indices_and_cleans_temporary_files(self) -> None:
        capture = self.make_capture(frames=5)
        observed_calls = []
        runner = Mock(
            side_effect=lambda *args, **kwargs: (
                observed_calls.append((args, kwargs, args[0].exists()))
                or self.make_assessment(
                    detections=[
                        {
                            "class_id": 0,
                            "class_name": "person",
                            "confidence": 0.9,
                            "box_xyxy": [1, 2, 3, 4],
                        }
                    ]
                )
            )
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            outcome = self.run_video(capture, runner)

        self.assertEqual(
            (
                outcome.sampled_frames,
                outcome.verified_frames,
                outcome.review_required_frames,
                outcome.failed_frames,
            ),
            (3, 3, 0, 0),
        )
        self.assertEqual(runner.call_count, 3)
        self.assertEqual(
            [call.kwargs["configuration_metadata"]["sampled_frame_index"]
             for call in runner.call_args_list],
            [0, 2, 4],
        )
        self.assertEqual(
            [call.kwargs["configuration_metadata"]["sampled_time_seconds"]
             for call in runner.call_args_list],
            [0.0, 1.0, 2.0],
        )
        self.assertTrue(all(call[2] for call in observed_calls))
        self.assertTrue(all(not call.args[0].exists() for call in runner.call_args_list))
        self.assertEqual(
            [call.args[1] for call in runner.call_args_list],
            ["video-model"] * 3,
        )
        self.assertTrue(
            all(call.kwargs["provenance_image_path"] == self.video_path
                for call in runner.call_args_list)
        )
        self.assertTrue(
            all(call.kwargs["database_settings"] is None
                for call in runner.call_args_list)
        )
        self.assertTrue(
            all(call.kwargs["generate_report"] is False
                for call in runner.call_args_list)
        )
        self.assertEqual(len(list((self.root / "reports").glob("*.pdf"))), 1)
        self.assertEqual(len(list((self.root / "reports").glob("*.json"))), 1)
        report_data = json.loads(
            outcome.sidecar_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            [frame["frame_index"] for frame in report_data["frames"]],
            [0, 2, 4],
        )
        self.assertEqual(report_data["video"]["filename"], self.video_path.name)
        self.assertEqual(report_data["video"]["sampling_interval_seconds"], 1.0)
        self.assertEqual(report_data["ai_results"]["detection_count"], 3)
        self.assertEqual(
            report_data["ai_results"]["object_class_counts"],
            {"person": 3},
        )
        self.assertTrue(outcome.report_path.read_bytes().startswith(b"%PDF-"))
        self.assertIn("Frame 2 at 1.000s: VERIFIED", output.getvalue())

    def test_continues_after_frame_assessment_failure_and_summarizes(self) -> None:
        capture = self.make_capture(frames=5)
        runner = Mock(
            side_effect=[
                RuntimeError("frame inference failed"),
                self.make_assessment(
                    "REVIEW REQUIRED",
                    findings=["quality warning"],
                    inference_succeeded=False,
                ),
                self.make_assessment(),
            ]
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            outcome = self.run_video(capture, runner)

        self.assertEqual(
            (
                outcome.sampled_frames,
                outcome.verified_frames,
                outcome.review_required_frames,
                outcome.failed_frames,
            ),
            (3, 1, 1, 1),
        )
        self.assertEqual(runner.call_count, 3)
        self.assertIn("Frame 0 at 0.000s: FAILED", output.getvalue())
        self.assertIn("Frame 2 at 1.000s: REVIEW REQUIRED", output.getvalue())
        self.assertNotIn("PDF report:", output.getvalue())
        self.assertEqual(len(list((self.root / "reports").glob("*.pdf"))), 1)
        report_data = json.loads(outcome.sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [frame["status"] for frame in report_data["frames"]],
            ["FAILED", "REVIEW REQUIRED", "VERIFIED"],
        )
        self.assertEqual(
            report_data["frames"][0]["findings"],
            ["frame inference failed"],
        )

    def test_decodes_a_generated_tiny_video(self) -> None:
        generated_video = self.root / "generated.avi"
        writer = cv2.VideoWriter(
            str(generated_video),
            cv2.VideoWriter_fourcc(*"MJPG"),
            2.0,
            (32, 24),
        )
        if not writer.isOpened():
            writer.release()
            self.skipTest("OpenCV MJPG video writer is unavailable.")
        for value in (0, 40, 80, 120, 160):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
        writer.release()

        runner = Mock(
            return_value=self.make_assessment()
        )
        outcome = run_video_assessments(
            generated_video,
            interval_seconds=1,
            model_id="video-model",
            registry_path=self.root / "registry.json",
            report_directory=self.root / "reports",
            confidence=0.4,
            anomaly_thresholds=AnomalyThresholds(),
            project_context=self.context,
            assessment_runner=runner,
        )

        self.assertEqual(outcome.sampled_frames, 3)
        self.assertEqual(outcome.verified_frames, 3)
        self.assertEqual(runner.call_count, 3)
        self.assertEqual(
            [
                call.kwargs["configuration_metadata"]["sampled_frame_index"]
                for call in runner.call_args_list
            ],
            [0, 2, 4],
        )
        self.assertEqual(len(list((self.root / "reports").glob("*.pdf"))), 1)
        self.assertEqual(len(list((self.root / "reports").glob("*.json"))), 1)

    def test_invalid_video_open_frame_rate_and_empty_stream_fail(self) -> None:
        with self.assertRaisesRegex(VideoAssessmentError, "does not exist"):
            run_video_assessments(
                self.root / "missing.mp4",
                interval_seconds=1,
                model_id="video-model",
                registry_path=self.root / "registry.json",
                report_directory=self.root / "reports",
                confidence=0.4,
                anomaly_thresholds=AnomalyThresholds(),
                project_context=self.context,
            )

        for capture, expected_message in (
            (self.make_capture(opened=False), "could not be opened"),
            (self.make_capture(frame_rate=0, frames=1), "frame rate"),
            (self.make_capture(frames=0), "no readable frames"),
        ):
            with self.subTest(error=expected_message):
                with self.assertRaisesRegex(VideoAssessmentError, expected_message):
                    self.run_video(capture, Mock())
                capture.release.assert_called_once()

    def test_cli_parses_options_and_returns_nonzero_for_failed_frame(self) -> None:
        arguments = build_parser().parse_args(
            [
                "assess-video",
                "--id",
                "video-model",
                "--video",
                "source.mp4",
                "--interval-seconds",
                "2.5",
                "--conf",
                "0.6",
                "--registry",
                "custom.json",
                "--project-config",
                str(self.config_path),
            ]
        )
        self.assertEqual(arguments.interval_seconds, 2.5)

        logger = Mock()
        output = io.StringIO()
        with (
            patch("assurance.configure_logging", return_value=logger),
            patch(
                "assurance.load_inference_settings",
                return_value=(AnomalyThresholds(), 0.25),
            ),
            patch(
                "assurance.run_video_assessments",
                return_value=VideoAssessmentOutcome(
                    sampled_frames=2,
                    verified_frames=1,
                    review_required_frames=0,
                    failed_frames=1,
                    report_path=self.root / "reports" / "video.pdf",
                    sidecar_path=self.root / "reports" / "video.json",
                ),
            ) as video_runner,
            contextlib.redirect_stdout(output),
        ):
            exit_code = main(
                [
                    "assess-video",
                    "--id",
                    "video-model",
                    "--video",
                    "source.mp4",
                    "--interval-seconds",
                    "2.5",
                    "--conf",
                    "0.6",
                    "--registry",
                    "custom.json",
                    "--project-config",
                    str(self.config_path),
                ]
            )

        self.assertEqual(exit_code, 2)
        video_runner.assert_called_once()
        self.assertEqual(video_runner.call_args.args[0], self.video_path)
        self.assertEqual(video_runner.call_args.kwargs["model_id"], "video-model")
        self.assertEqual(video_runner.call_args.kwargs["interval_seconds"], 2.5)
        self.assertEqual(
            video_runner.call_args.kwargs["registry_path"],
            self.root / "custom.json",
        )
        self.assertIn(
            "sampled=2, successful/verified=1, review-required=0, failed=1",
            output.getvalue(),
        )
        self.assertIn("Consolidated PDF report:", output.getvalue())

    def test_cli_rejects_nonpositive_interval(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "assess-video",
                    "--id",
                    "video-model",
                    "--video",
                    "source.mp4",
                    "--interval-seconds",
                    "0",
                ]
            )


if __name__ == "__main__":
    unittest.main()
