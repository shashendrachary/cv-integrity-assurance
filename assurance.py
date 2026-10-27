"""Command-line entry point for CV-Integrity-Assurance."""

import argparse
import logging
import math
import os
import stat
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2

from src.anomaly.checks import AnomalyThresholds, load_inference_settings
from src.demos import run_data_integrity_demo, run_model_tampering_demo
from src.integrity.data_checks import (
    ImageDependencyError,
    SUPPORTED_IMAGE_EXTENSIONS,
    format_data_check,
    scan_dataset,
    sha256_file,
)
from src.integrity.image_baselines import (
    ImageRegistryError,
    format_image_verification,
    register_image,
    verify_image,
)
from src.database.postgres import (
    ConnectionSettings,
    DatabaseNotConfigured,
    DatabaseOperationError,
    initialize_provenance_table,
    load_database_settings,
    migrate_project_scope_schema,
)
from src.model.integrity import (
    ModelRegistryError,
    format_verification,
    register_model,
    verify_model,
)
from src.provenance.records import create_provenance_record, insert_provenance_record
from src.provenance.workflow import AssessmentResult, run_assessment
from src.project_context import ProjectContext, ProjectContextError, load_project_context
from src.reports.pdf_report import (
    BatchImageReportItem,
    BatchReportSummary,
    ReportGenerationError,
    VideoFrameReportItem,
    VideoReportSummary,
    generate_batch_pdf_report,
    generate_video_pdf_report,
)
from src.vision.inference import format_inference_result, run_inference


PROJECT_NAME = "CV-Integrity-Assurance"
PROJECT_TITLE = (
    "Trustworthy Computer Vision Integrity Assurance for Data, Models "
    "and Inference Outputs in Multi-Contributor Pipelines"
)
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "logs" / "assurance.log"

IMPLEMENTED_CAPABILITIES = (
    "Dataset integrity checks",
    "Model baseline registration and integrity verification",
    "Local YOLO object detection with image-quality review checks",
    "PostgreSQL inference provenance evidence",
    "End-to-end assessment with local PDF reports",
)

PLANNED_CAPABILITIES: tuple[str, ...] = ()


class BatchImageDiscoveryError(ValueError):
    """Raised when a batch input directory cannot be scanned usefully."""


class VideoAssessmentError(ValueError):
    """Raised when a video cannot be decoded or sampled safely."""


@dataclass(frozen=True)
class BatchAssessmentEntry:
    """Outcome for one attempted image in a batch."""

    image_path: Path
    assessment: AssessmentResult | None = None
    error: str | None = None

    @property
    def status(self) -> str:
        if self.assessment is None:
            return "FAILED"
        return self.assessment.overall_status


@dataclass(frozen=True)
class VideoAssessmentOutcome:
    """Frame totals and the one consolidated report generated for a video."""

    sampled_frames: int
    verified_frames: int
    review_required_frames: int
    failed_frames: int
    report_path: Path
    sidecar_path: Path


def validate_video_interval(interval_seconds: float) -> None:
    """Reject intervals that cannot represent a positive sampling period."""
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise VideoAssessmentError("Frame interval must be a finite number greater than zero.")


def parse_video_interval(value: str) -> float:
    """Parse a positive finite frame interval for argparse."""
    try:
        interval_seconds = float(value)
        validate_video_interval(interval_seconds)
    except (ValueError, VideoAssessmentError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return interval_seconds


def discover_batch_images(images_directory: Path) -> list[Path]:
    """Return supported images recursively in deterministic path order."""
    try:
        directory_info = images_directory.stat()
    except FileNotFoundError as error:
        raise BatchImageDiscoveryError(
            f"Images directory does not exist: {images_directory}"
        ) from error
    except OSError as error:
        raise BatchImageDiscoveryError(
            f"Unable to access images directory {images_directory}: {error}"
        ) from error
    if not stat.S_ISDIR(directory_info.st_mode):
        raise BatchImageDiscoveryError(
            f"Images path is not a directory: {images_directory}"
        )

    images: list[Path] = []
    scan_errors: list[OSError] = []

    def record_scan_error(error: OSError) -> None:
        scan_errors.append(error)

    for current_directory, directories, filenames in os.walk(
        images_directory,
        onerror=record_scan_error,
    ):
        directories.sort()
        images.extend(
            Path(current_directory) / filename
            for filename in filenames
            if Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

    if scan_errors:
        raise BatchImageDiscoveryError(
            f"Unable to read all of images directory {images_directory}: "
            f"{scan_errors[0]}"
        )
    if not images:
        raise BatchImageDiscoveryError(
            f"No supported images found in directory: {images_directory}"
        )
    return sorted(images, key=lambda image: (str(image).casefold(), str(image)))


def run_batch_assessments(
    image_paths: list[Path],
    *,
    model_id: str,
    registry_path: Path,
    report_directory: Path,
    confidence: float,
    anomaly_thresholds: AnomalyThresholds,
    database_settings: ConnectionSettings | None,
    database_note: str | None,
    project_context: ProjectContext,
    assessment_runner: Callable[..., AssessmentResult] | None = None,
    logger: logging.Logger | None = None,
    image_registry_path: Path | None = None,
) -> list[BatchAssessmentEntry]:
    """Assess each image without creating individual PDF reports."""
    if not image_paths:
        raise ValueError("No images were supplied for batch assessment.")
    if assessment_runner is None:
        assessment_runner = run_assessment

    entries: list[BatchAssessmentEntry] = []
    for image_path in image_paths:
        try:
            assessment = assessment_runner(
                image_path,
                model_id,
                registry_path,
                report_directory,
                confidence=confidence,
                anomaly_thresholds=anomaly_thresholds,
                database_settings=database_settings,
                database_note=database_note,
                project_root=project_context.base_path,
                image_registry_path=image_registry_path,
                project_context=project_context,
                generate_report=False,
            )
        except Exception as error:
            if logger is not None:
                logger.error("Batch assessment failed for %s: %s", image_path, error)
            print(f"Image: {image_path}\nStatus: FAILED\nError: {error}")
            entries.append(BatchAssessmentEntry(image_path, error=str(error)))
            continue

        print(f"Image: {image_path}\nStatus: {assessment.overall_status}")
        entries.append(BatchAssessmentEntry(image_path, assessment=assessment))

    return entries


def build_batch_report_data(
    image_paths: list[Path],
    entries: list[BatchAssessmentEntry],
    model_id: str,
) -> tuple[list[BatchImageReportItem], BatchReportSummary]:
    """Convert per-image outcomes into compact report rows and accurate totals."""
    if not image_paths:
        raise ValueError("No images were discovered for batch assessment.")
    if len(entries) != len(image_paths):
        raise ValueError("Every discovered image must have one batch result.")

    hashes: list[str | None] = []
    image_properties: list[tuple[int | None, int | None, str | None]] = []
    for entry in entries:
        record = entry.assessment.provenance if entry.assessment is not None else None
        image_hash = record.image_sha256 if record is not None else None
        if image_hash is None:
            try:
                image_hash = sha256_file(entry.image_path)
            except OSError:
                image_hash = None
        hashes.append(image_hash)

        width = height = None
        image_format = None
        if entry.assessment is not None:
            metrics = entry.assessment.provenance.image_metrics
            width = metrics.get("width")
            height = metrics.get("height")
        try:
            from PIL import Image

            with Image.open(entry.image_path) as image:
                image_format = image.format
                width = width if width is not None else image.width
                height = height if height is not None else image.height
        except (ImportError, OSError, ValueError):
            pass
        image_properties.append((width, height, image_format))

    hash_counts = Counter(image_hash for image_hash in hashes if image_hash is not None)
    report_items: list[BatchImageReportItem] = []
    for entry, image_hash, properties in zip(entries, hashes, image_properties):
        width, height, image_format = properties
        assessment = entry.assessment
        if assessment is None:
            report_items.append(
                BatchImageReportItem(
                    filename=entry.image_path.name,
                    image_sha256=image_hash,
                    width=width,
                    height=height,
                    image_format=image_format,
                    duplicate_status=(
                        f"Duplicate ({hash_counts[image_hash]} copies)"
                        if image_hash is not None and hash_counts[image_hash] > 1
                        else "Unique" if image_hash is not None else "N/A"
                    ),
                    quality_flags=(entry.error or "Assessment failed",),
                    status="FAILED",
                    model_verification_status="N/A",
                    inference_succeeded=False,
                    reasons=(entry.error or "Assessment failed",),
                    detections=(),
                )
            )
            continue

        inference = assessment.inference
        record = assessment.provenance
        reasons = tuple(dict.fromkeys([*record.reasons, *inference.reasons]))
        flags = list(inference.image_analysis.reasons)
        image_verification = getattr(inference, "image_verification", None)
        if (
            image_verification is not None
            and image_verification.status != "VERIFIED"
        ):
            flags.append(
                f"Image integrity: {image_verification.status}"
            )
        if inference.model_verification.status != "VERIFIED":
            flags.append(
                f"Model verification: {inference.model_verification.status}"
            )
        flags.extend(
            reason for reason in inference.reasons if reason not in flags
        )
        report_items.append(
            BatchImageReportItem(
                filename=entry.image_path.name,
                image_sha256=image_hash,
                width=width,
                height=height,
                image_format=image_format,
                duplicate_status=(
                    f"Duplicate ({hash_counts[image_hash]} copies)"
                    if image_hash is not None and hash_counts[image_hash] > 1
                    else "Unique" if image_hash is not None else "N/A"
                ),
                quality_flags=tuple(flags),
                status=assessment.overall_status,
                model_verification_status=inference.model_verification.status,
                inference_succeeded=inference.inference_succeeded,
                reasons=reasons,
                detections=tuple(record.detections),
            )
        )

    successful_count = sum(entry.status == "VERIFIED" for entry in entries)
    review_count = sum(entry.status == "REVIEW REQUIRED" for entry in entries)
    failure_count = len(entries) - successful_count - review_count
    inference_failure_count = sum(
        entry.assessment is None
        or not entry.assessment.inference.inference_succeeded
        for entry in entries
    )
    model_statuses = sorted(
        {
            entry.assessment.inference.model_verification.status
            for entry in entries
            if entry.assessment is not None
        }
    )
    model_status = (
        model_statuses[0]
        if len(model_statuses) == 1
        else f"MIXED ({', '.join(model_statuses)})"
        if model_statuses
        else "N/A"
    )
    report_summary = BatchReportSummary(
        discovered=len(image_paths),
        processed=len(entries),
        successful=successful_count,
        review_required=review_count,
        failures=failure_count,
        inference_failures=inference_failure_count,
        model_id=model_id,
        model_verification_status=model_status,
    )
    return report_items, report_summary


def run_video_assessments(
    video_path: Path,
    *,
    interval_seconds: float,
    model_id: str,
    registry_path: Path,
    report_directory: Path,
    confidence: float,
    anomaly_thresholds: AnomalyThresholds,
    project_context: ProjectContext,
    assessment_runner: Callable[..., AssessmentResult] | None = None,
    capture_factory: Callable[[str], Any] | None = None,
    frame_writer: Callable[[str, Any], bool] | None = None,
    report_generator: Callable[
        [list[VideoFrameReportItem], VideoReportSummary, Path],
        tuple[Path, Path],
    ] | None = None,
    logger: logging.Logger | None = None,
    image_registry_path: Path | None = None,
) -> VideoAssessmentOutcome:
    """Assess sampled frames, then write one PDF and JSON report for the video."""
    validate_video_interval(interval_seconds)
    if not video_path.is_file():
        raise VideoAssessmentError(
            f"Video file does not exist or is not a file: {video_path}"
        )

    assessment_runner = assessment_runner or run_assessment
    capture_factory = capture_factory or cv2.VideoCapture
    frame_writer = frame_writer or cv2.imwrite
    report_generator = report_generator or generate_video_pdf_report

    try:
        capture = capture_factory(str(video_path))
    except Exception as error:
        raise VideoAssessmentError(
            f"Unable to open video {video_path}: {error}"
        ) from error

    frame_results: list[VideoFrameReportItem] = []
    decoded_count = 0
    try:
        if not capture.isOpened():
            raise VideoAssessmentError(
                f"Video could not be opened; the file may be invalid or use an "
                f"unsupported codec: {video_path}"
            )

        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise VideoAssessmentError(
                f"Video has an invalid or unsupported frame rate: {video_path}"
            )

        next_sample_time = 0.0
        with tempfile.TemporaryDirectory(
            prefix="assurance-video-frames-"
        ) as temporary_directory:
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break

                frame_time = decoded_count / frame_rate
                if frame_time + 1e-9 >= next_sample_time:
                    frame_index = decoded_count
                    frame_path = (
                        Path(temporary_directory) / f"frame-{frame_index:08d}.png"
                    )
                    try:
                        if not frame_writer(str(frame_path), frame):
                            raise VideoAssessmentError(
                                f"Could not write sampled frame {frame_index}."
                            )
                        assessment = assessment_runner(
                            frame_path,
                            model_id,
                            registry_path,
                            report_directory,
                            confidence=confidence,
                            anomaly_thresholds=anomaly_thresholds,
                            database_settings=None,
                            database_note=(
                                "skipped (video assessment does not connect to PostgreSQL)"
                            ),
                            project_root=project_context.base_path,
                            image_registry_path=image_registry_path,
                            project_context=project_context,
                            provenance_image_path=video_path,
                            configuration_metadata={
                                "source_type": "video_frame",
                                "video_path": str(video_path),
                                "sampled_frame_index": frame_index,
                                "sampled_time_seconds": frame_time,
                                "sampling_interval_seconds": interval_seconds,
                                "image_sha256_scope": "temporary encoded frame",
                            },
                            generate_report=False,
                        )
                        inference = assessment.inference
                        record = assessment.provenance
                        findings = tuple(
                            dict.fromkeys(
                                [
                                    *record.reasons,
                                    *inference.reasons,
                                    *inference.image_analysis.reasons,
                                ]
                            )
                        )
                        frame_results.append(
                            VideoFrameReportItem(
                                frame_index=frame_index,
                                time_seconds=frame_time,
                                status=assessment.overall_status,
                                detection_count=len(inference.detections),
                                model_verification_status=(
                                    inference.model_verification.status
                                ),
                                findings=findings,
                                detections=tuple(record.detections),
                            )
                        )
                        print(
                            f"Frame {frame_index} at {frame_time:.3f}s: "
                            f"{assessment.overall_status}"
                        )
                    except Exception as error:
                        if logger is not None:
                            logger.error(
                                "Video assessment failed for frame %d at %.3fs: %s",
                                frame_index,
                                frame_time,
                                error,
                            )
                        print(
                            f"Frame {frame_index} at {frame_time:.3f}s: FAILED\n"
                            f"Error: {error}"
                        )
                        frame_results.append(
                            VideoFrameReportItem(
                                frame_index=frame_index,
                                time_seconds=frame_time,
                                status="FAILED",
                                detection_count=0,
                                model_verification_status="N/A",
                                findings=(str(error),),
                                detections=(),
                            )
                        )
                    finally:
                        try:
                            frame_path.unlink(missing_ok=True)
                        except OSError as error:
                            if logger is not None:
                                logger.error(
                                    "Could not remove temporary frame %s: %s",
                                    frame_path,
                                    error,
                                )
                            if frame_results and frame_results[-1].frame_index == frame_index:
                                previous = frame_results[-1]
                                if previous.status != "FAILED":
                                    frame_results[-1] = VideoFrameReportItem(
                                        frame_index=previous.frame_index,
                                        time_seconds=previous.time_seconds,
                                        status="FAILED",
                                        detection_count=previous.detection_count,
                                        model_verification_status=(
                                            previous.model_verification_status
                                        ),
                                        findings=(
                                            *previous.findings,
                                            f"Temporary frame cleanup failed: {error}",
                                        ),
                                        detections=previous.detections,
                                    )

                    while next_sample_time <= frame_time + 1e-9:
                        next_sample_time += interval_seconds

                decoded_count += 1

        if decoded_count == 0:
            raise VideoAssessmentError(
                f"Video yielded no readable frames; it may be empty, corrupt, or "
                f"use an unsupported codec: {video_path}"
            )

        verified_count = sum(
            result.status == "VERIFIED" for result in frame_results
        )
        review_count = sum(
            result.status == "REVIEW REQUIRED" for result in frame_results
        )
        failed_count = len(frame_results) - verified_count - review_count
        model_statuses = sorted(
            {
                result.model_verification_status
                for result in frame_results
                if result.model_verification_status != "N/A"
            }
        )
        model_status = (
            model_statuses[0]
            if len(model_statuses) == 1
            else f"MIXED ({', '.join(model_statuses)})"
            if model_statuses
            else "N/A"
        )
        report_path, sidecar_path = report_generator(
            frame_results,
            VideoReportSummary(
                video_filename=video_path.name,
                video_path=str(video_path),
                interval_seconds=interval_seconds,
                model_id=model_id,
                sampled_frames=len(frame_results),
                verified_frames=verified_count,
                review_required_frames=review_count,
                failed_frames=failed_count,
                model_verification_status=model_status,
            ),
            report_directory,
        )
        return VideoAssessmentOutcome(
            sampled_frames=len(frame_results),
            verified_frames=verified_count,
            review_required_frames=review_count,
            failed_frames=failed_count,
            report_path=report_path,
            sidecar_path=sidecar_path,
        )
    except cv2.error as error:
        raise VideoAssessmentError(
            f"OpenCV could not decode video {video_path}: {error}"
        ) from error
    finally:
        capture.release()


def configure_logging() -> logging.Logger:
    """Use file logging when possible, without blocking CLI startup if it fails."""
    logger = logging.getLogger("integrity_assurance")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    except OSError as error:
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
        logger.warning("File logging is unavailable; using stderr: %s", error)
    else:
        logger.addHandler(handler)

    for active_handler in logger.handlers:
        active_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
    return logger


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="assurance.py",
        description=f"Command-line foundation for {PROJECT_NAME}.",
    )
    commands = parser.add_subparsers(dest="command")

    def add_project_config_argument(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--project-config",
            type=Path,
            help=(
                "select a project configuration (relative paths use its directory; "
                "default: project config.yaml)"
            ),
        )

    data_parser = commands.add_parser(
        "check-data",
        help="validate dataset images and YOLO labels",
        description="Check image integrity, duplicates, YOLO labels, and file pairings.",
    )
    data_parser.add_argument(
        "--images-dir",
        type=Path,
        help="override the configured image directory",
    )
    data_parser.add_argument(
        "--labels-dir",
        type=Path,
        help="override the configured label directory",
    )
    add_project_config_argument(data_parser)
    register_parser = commands.add_parser(
        "register-model",
        help="register a model file's SHA-256 baseline",
        description="Create or explicitly replace a local model integrity baseline.",
    )
    register_parser.add_argument("--id", required=True, help="unique model identifier")
    register_parser.add_argument("--path", required=True, type=Path, help="model file path")
    register_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    register_parser.add_argument(
        "--replace",
        action="store_true",
        help="explicitly replace an existing trusted baseline",
    )
    add_project_config_argument(register_parser)
    register_image_parser = commands.add_parser(
        "register-image",
        help="register an image file's trusted SHA-256 baseline",
        description="Create or explicitly replace a local trusted image baseline.",
    )
    register_image_parser.add_argument("--image", required=True, type=Path)
    register_image_parser.add_argument("--registry", type=Path)
    register_image_parser.add_argument("--replace", action="store_true")
    add_project_config_argument(register_image_parser)
    verify_parser = commands.add_parser(
        "verify-model",
        help="verify a registered model file against its baseline",
        description="Compare the current model SHA-256 with its registered baseline.",
    )
    verify_parser.add_argument("--id", required=True, help="registered model identifier")
    verify_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(verify_parser)
    verify_image_parser = commands.add_parser(
        "verify-image",
        help="verify an image against its trusted SHA-256 baseline",
        description="Compare an image's current SHA-256 with its trusted baseline.",
    )
    verify_image_parser.add_argument("--image", required=True, type=Path)
    verify_image_parser.add_argument("--registry", type=Path)
    add_project_config_argument(verify_image_parser)
    inference_parser = commands.add_parser(
        "infer",
        help="run integrity-gated local YOLO inference",
        description="Analyze an image and run CPU detection only with a verified model.",
    )
    inference_parser.add_argument("--id", required=True, help="registered model identifier")
    inference_parser.add_argument("--image", required=True, type=Path, help="input image path")
    inference_parser.add_argument(
        "--conf",
        type=float,
        help="confidence threshold from 0 to 1 (default: config.yaml setting)",
    )
    inference_parser.add_argument(
        "--image-registry", type=Path, help="override the configured image baseline registry"
    )
    inference_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(inference_parser)
    batch_parser = commands.add_parser(
        "assess-batch",
        help="assess every supported image in a directory",
        description=(
            "Run integrity-gated assessment, provenance persistence, and PDF "
            "report generation for each supported image in a directory."
        ),
    )
    batch_parser.add_argument("--id", required=True, help="registered model identifier")
    batch_parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="directory of images to assess recursively",
    )
    batch_parser.add_argument(
        "--image-registry", type=Path, help="override the configured image baseline registry"
    )
    batch_parser.add_argument(
        "--conf",
        type=float,
        help="confidence threshold from 0 to 1 (default: config.yaml setting)",
    )
    batch_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(batch_parser)
    video_parser = commands.add_parser(
        "assess-video",
        help="assess video frames sampled at a time interval",
        description=(
            "Sample video frames sequentially and run the existing integrity-gated "
            "assessment workflow for each sampled frame."
        ),
    )
    video_parser.add_argument("--id", required=True, help="registered model identifier")
    video_parser.add_argument("--video", type=Path, required=True, help="input video path")
    video_parser.add_argument(
        "--interval-seconds",
        type=parse_video_interval,
        default=1.0,
        help="time between sampled frames in seconds (default: 1.0)",
    )
    video_parser.add_argument(
        "--image-registry", type=Path, help="override the configured image baseline registry"
    )
    video_parser.add_argument(
        "--conf",
        type=float,
        help="confidence threshold from 0 to 1 (default: config.yaml setting)",
    )
    video_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(video_parser)
    commands.add_parser(
        "init-db",
        help="create the legacy PostgreSQL provenance table if absent",
        description=(
            "Explicitly create the legacy provenance table if absent; this does "
            "not run the project-scope migration."
        ),
    )
    commands.add_parser(
        "migrate-db",
        help="explicitly migrate PostgreSQL provenance to project-scoped schema",
        description=(
            "Run the versioned project-scope migration. Back up the database first."
        ),
    )
    assessment_parser = commands.add_parser(
        "assess",
        help="run inference, record available evidence, and generate a PDF report",
        description="Run a complete local image/model assurance assessment.",
    )
    assessment_parser.add_argument("--id", required=True, help="registered model identifier")
    assessment_parser.add_argument("--image", required=True, type=Path, help="input image path")
    assessment_parser.add_argument(
        "--conf",
        type=float,
        help="confidence threshold from 0 to 1 (default: config.yaml setting)",
    )
    assessment_parser.add_argument(
        "--image-registry", type=Path, help="override the configured image baseline registry"
    )
    assessment_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(assessment_parser)
    tamper_demo_parser = commands.add_parser(
        "demo-model-tampering",
        help="safely demonstrate model hash mismatch using a temporary copy",
    )
    tamper_demo_parser.add_argument(
        "--id",
        default="yolo11n",
        help="registered source model identifier (default: yolo11n)",
    )
    tamper_demo_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(tamper_demo_parser)
    commands.add_parser(
        "demo-data-integrity",
        help="demonstrate image and YOLO-label findings with temporary files",
    )
    return parser


def _project_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root unless already absolute."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_cli_project_context(project_config: Path | None) -> ProjectContext:
    configuration_path = (
        _project_path(project_config)
        if project_config is not None
        else PROJECT_ROOT / "config.yaml"
    )
    return load_project_context(configuration_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the foundation status, dataset check, or model-integrity command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    logger = configure_logging()
    project_context: ProjectContext | None = None
    if arguments.command in {
        "check-data",
        "register-model",
        "verify-model",
        "register-image",
        "verify-image",
        "infer",
        "assess",
        "assess-batch",
        "assess-video",
        "demo-model-tampering",
    }:
        try:
            project_context = _load_cli_project_context(arguments.project_config)
        except ProjectContextError as error:
            logger.error("Project configuration could not be loaded: %s", error)
            print(f"Project configuration failed: {error}", file=sys.stderr)
            return 2
        print(
            "Project context: "
            f"{project_context.organization_id}/{project_context.project_id}"
        )

    if arguments.command in {"init-db", "migrate-db"}:
        try:
            settings = load_database_settings(PROJECT_ROOT / ".env")
            if settings is None:
                raise DatabaseNotConfigured(
                    "PostgreSQL settings are not configured. Set POSTGRES_DB and "
                    "POSTGRES_USER in the environment or project .env file."
                )
            if arguments.command == "migrate-db":
                version = migrate_project_scope_schema(settings)
                print(
                    "Project-scoped PostgreSQL schema is at version "
                    f"{version} (existing provenance preserved)."
                )
                return 0
            initialize_provenance_table(settings)
        except (DatabaseNotConfigured, DatabaseOperationError) as error:
            print(f"Database initialization failed: {error}", file=sys.stderr)
            return 2
        print("PostgreSQL provenance table initialized (existing records preserved).")
        return 0

    if arguments.command == "demo-data-integrity":
        run_data_integrity_demo()
        return 0

    if arguments.command == "demo-model-tampering":
        assert project_context is not None
        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            run_model_tampering_demo(
                arguments.id,
                registry_path,
                project_root=project_context.base_path,
            )
        except (OSError, ValueError, ModelRegistryError, RuntimeError) as error:
            print(f"Model tampering demo failed: {error}", file=sys.stderr)
            return 2
        return 0

    if arguments.command in {"register-image", "verify-image"}:
        assert project_context is not None
        registry_path = (
            project_context.resolve_path(arguments.registry)
            if arguments.registry is not None
            else project_context.image_registry
        )
        image_path = project_context.resolve_path(arguments.image)
        try:
            if arguments.command == "register-image":
                registration = register_image(
                    image_path, registry_path, replace=arguments.replace
                )
                print(f"Image: {registration['file_path']}")
                print(f"Expected SHA-256: {registration['expected_hash']}")
                print(f"Registered at: {registration['registered_at']}")
                return 0
            result = verify_image(image_path, registry_path)
            print(format_image_verification(result))
            return 0 if result.status == "VERIFIED" else 2
        except (OSError, ValueError, ImageRegistryError) as error:
            logger.error("Image integrity operation failed: %s", error)
            print(f"Image integrity operation failed: {error}", file=sys.stderr)
            return 2

    if arguments.command in {"register-model", "verify-model"}:
        assert project_context is not None
        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            if arguments.command == "register-model":
                registration = register_model(
                    arguments.id,
                    project_context.resolve_path(arguments.path),
                    registry_path,
                    replace=arguments.replace,
                )
                logger.info("Registered model baseline: %s", arguments.id)
                print(f"Model ID: {registration['model_id']}")
                print(f"File: {registration['file_path']}")
                print(f"Expected SHA-256: {registration['expected_hash']}")
                print(f"Registered at: {registration['registered_at']}")
                print(
                    "Note: registration establishes a local baseline; it does not "
                    "prove the original model is authentic."
                )
                return 0

            result = verify_model(
                arguments.id,
                registry_path,
                project_root=project_context.base_path,
            )
            print(format_verification(result))
            logger.info("Model verification %s: %s", result.status, arguments.id)
            return 0 if result.status == "VERIFIED" else 2
        except (OSError, ValueError, ModelRegistryError) as error:
            logger.error("Model operation could not be completed: %s", error)
            print(f"Model operation failed: {error}", file=sys.stderr)
            return 2

    if arguments.command == "infer":
        assert project_context is not None
        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            image_registry_path = (
                project_context.resolve_path(arguments.image_registry)
                if arguments.image_registry is not None
                else project_context.image_registry
            )
            anomaly_thresholds, configured_confidence = load_inference_settings(
                project_context.configuration_path
            )
            result = run_inference(
                project_context.resolve_path(arguments.image),
                arguments.id,
                registry_path,
                confidence=(
                    arguments.conf
                    if arguments.conf is not None
                    else configured_confidence
                ),
                anomaly_thresholds=anomaly_thresholds,
                project_root=project_context.base_path,
                image_registry_path=image_registry_path,
            )
        except (OSError, ValueError, ModelRegistryError) as error:
            logger.error("Inference could not be started: %s", error)
            print(f"Inference failed: {error}", file=sys.stderr)
            return 2

        print(format_inference_result(result))
        record = create_provenance_record(
            project_context.resolve_path(arguments.image),
            result,
            {
                "confidence": result.confidence,
                "device": "cpu",
                "image_integrity": {
                    "status": result.image_verification.status,
                    "expected_sha256": result.image_verification.expected_hash,
                    "actual_sha256": result.image_verification.actual_hash,
                    "reason": result.image_verification.reason,
                },
                "anomaly_thresholds": {
                    "minimum_width": anomaly_thresholds.minimum_width,
                    "minimum_height": anomaly_thresholds.minimum_height,
                    "maximum_width": anomaly_thresholds.maximum_width,
                    "maximum_height": anomaly_thresholds.maximum_height,
                    "minimum_aspect_ratio": anomaly_thresholds.minimum_aspect_ratio,
                    "maximum_aspect_ratio": anomaly_thresholds.maximum_aspect_ratio,
                    "minimum_mean_brightness": anomaly_thresholds.minimum_mean_brightness,
                    "maximum_mean_brightness": anomaly_thresholds.maximum_mean_brightness,
                    "minimum_laplacian_variance": anomaly_thresholds.minimum_laplacian_variance,
                },
            },
            project_context,
        )
        print(f"Provenance run ID: {record.run_id}")
        try:
            database_settings = load_database_settings(PROJECT_ROOT / ".env")
            if database_settings is None:
                print(
                    "Evidence persistence: skipped (PostgreSQL is not configured). "
                    "Set POSTGRES_DB and POSTGRES_USER, then run init-db and "
                    "explicitly migrate-db for project-scoped persistence."
                )
            else:
                insert_provenance_record(record, database_settings)
                print("Evidence persistence: saved to PostgreSQL.")
        except (DatabaseNotConfigured, DatabaseOperationError) as error:
            logger.warning("Provenance evidence was not persisted: %s", error)
            print(
                "Evidence persistence: not saved; PostgreSQL is unavailable or "
                f"misconfigured ({error})."
            )
        logger.info(
            "Inference run status %s for model %s",
            result.status,
            arguments.id,
        )
        return 0 if result.status == "VERIFIED" else 2

    if arguments.command == "assess":
        assert project_context is not None
        image_path = project_context.resolve_path(arguments.image)
        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            image_registry_path = (
                project_context.resolve_path(arguments.image_registry)
                if arguments.image_registry is not None
                else project_context.image_registry
            )
            report_directory = project_context.reports_directory
            anomaly_thresholds, configured_confidence = load_inference_settings(
                project_context.configuration_path
            )
            confidence = (
                arguments.conf
                if arguments.conf is not None
                else configured_confidence
            )
            database_note = None
            try:
                database_settings = load_database_settings(PROJECT_ROOT / ".env")
            except DatabaseNotConfigured as error:
                database_settings = None
                database_note = (
                    f"skipped (database configuration unavailable: {error})"
                )

            assessment = run_assessment(
                image_path,
                arguments.id,
                registry_path,
                report_directory,
                confidence=confidence,
                anomaly_thresholds=anomaly_thresholds,
                database_settings=database_settings,
                database_note=database_note,
                project_root=project_context.base_path,
                project_context=project_context,
                image_registry_path=image_registry_path,
            )
        except (
            OSError,
            ValueError,
            ModelRegistryError,
            ReportGenerationError,
        ) as error:
            logger.error("Assessment could not be completed: %s", error)
            print(f"Assessment failed: {error}", file=sys.stderr)
            return 2

        inference_summary = format_inference_result(assessment.inference).replace(
            "Run status:", "Inference status:", 1
        )
        print(inference_summary)
        print(f"Overall status: {assessment.overall_status}")
        print(f"Provenance run ID: {assessment.provenance.run_id}")
        print(f"PostgreSQL evidence: {assessment.persistence_status}")
        print(f"PDF report: {assessment.report_path}")
        logger.info(
            "Assessment %s completed with status %s",
            assessment.provenance.run_id,
            assessment.overall_status,
        )
        return 0 if assessment.overall_status == "VERIFIED" else 2

    if arguments.command == "assess-batch":
        assert project_context is not None
        images_directory = project_context.resolve_path(arguments.images_dir)
        try:
            image_paths = discover_batch_images(images_directory)
        except BatchImageDiscoveryError as error:
            logger.error("Batch image discovery failed: %s", error)
            print(f"Batch assessment failed: {error}", file=sys.stderr)
            return 2

        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            image_registry_path = (
                project_context.resolve_path(arguments.image_registry)
                if arguments.image_registry is not None
                else project_context.image_registry
            )
            anomaly_thresholds, configured_confidence = load_inference_settings(
                project_context.configuration_path
            )
            confidence = (
                arguments.conf
                if arguments.conf is not None
                else configured_confidence
            )
            database_note = None
            try:
                database_settings = load_database_settings(PROJECT_ROOT / ".env")
            except DatabaseNotConfigured as error:
                database_settings = None
                database_note = (
                    f"skipped (database configuration unavailable: {error})"
                )
        except (OSError, ValueError, DatabaseNotConfigured) as error:
            logger.error("Batch assessment could not be configured: %s", error)
            print(f"Batch assessment failed: {error}", file=sys.stderr)
            return 2

        try:
            entries = run_batch_assessments(
                image_paths,
                model_id=arguments.id,
                registry_path=registry_path,
                report_directory=project_context.reports_directory,
                confidence=confidence,
                anomaly_thresholds=anomaly_thresholds,
                database_settings=database_settings,
                database_note=database_note,
                project_context=project_context,
                logger=logger,
                image_registry_path=image_registry_path,
            )
            report_items, summary = build_batch_report_data(
                image_paths,
                entries,
                arguments.id,
            )
            pdf_path, sidecar_path = generate_batch_pdf_report(
                report_items,
                summary,
                project_context.reports_directory,
            )
        except (OSError, ValueError, ReportGenerationError) as error:
            logger.error("Consolidated batch report failed: %s", error)
            print(f"Batch report failed: {error}", file=sys.stderr)
            return 2

        print(
            f"Batch totals: discovered={summary.discovered}, "
            f"processed={summary.processed}, successful={summary.successful}, "
            f"review-required={summary.review_required}, failures={summary.failures}"
        )
        print(f"Consolidated PDF report: {pdf_path}")
        print(f"Detection details: {sidecar_path}")
        return (
            0
            if summary.review_required == 0 and summary.failures == 0
            else 2
        )

    if arguments.command == "assess-video":
        assert project_context is not None
        video_path = project_context.resolve_path(arguments.video)
        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            image_registry_path = (
                project_context.resolve_path(arguments.image_registry)
                if arguments.image_registry is not None
                else project_context.image_registry
            )
            anomaly_thresholds, configured_confidence = load_inference_settings(
                project_context.configuration_path
            )
            confidence = (
                arguments.conf
                if arguments.conf is not None
                else configured_confidence
            )
            outcome = run_video_assessments(
                video_path,
                interval_seconds=arguments.interval_seconds,
                model_id=arguments.id,
                registry_path=registry_path,
                report_directory=project_context.reports_directory,
                confidence=confidence,
                anomaly_thresholds=anomaly_thresholds,
                project_context=project_context,
                logger=logger,
                image_registry_path=image_registry_path,
            )
        except (OSError, ValueError, ModelRegistryError, ReportGenerationError) as error:
            logger.error("Video assessment could not be completed: %s", error)
            print(f"Video assessment failed: {error}", file=sys.stderr)
            return 2

        print(
            f"Video totals: sampled={outcome.sampled_frames}, "
            f"successful/verified={outcome.verified_frames}, "
            f"review-required={outcome.review_required_frames}, "
            f"failed={outcome.failed_frames}"
        )
        print(f"Consolidated PDF report: {outcome.report_path}")
        print(f"Frame details: {outcome.sidecar_path}")
        return (
            0
            if outcome.review_required_frames == 0 and outcome.failed_frames == 0
            else 2
        )

    if arguments.command == "check-data":
        assert project_context is not None
        try:
            result = scan_dataset(
                project_context.resolve_path(arguments.images_dir)
                if arguments.images_dir is not None
                else project_context.images_directory,
                project_context.resolve_path(arguments.labels_dir)
                if arguments.labels_dir is not None
                else project_context.labels_directory,
            )
        except (OSError, ValueError, ImageDependencyError) as error:
            logger.error("Data check could not be completed: %s", error)
            print(f"Data check failed: {error}", file=sys.stderr)
            return 2

        logger.info(
            "Data check completed: %d images, %d valid labels",
            result.images_examined,
            result.valid_label_files,
        )
        print(format_data_check(result))
        return 2 if result.operational_errors else 0

    logger.info("Foundation status requested")

    required_directories = (
        PROJECT_ROOT / "dataset" / "images",
        PROJECT_ROOT / "dataset" / "labels",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "reports",
        PROJECT_ROOT / "logs",
    )
    missing_directories = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in required_directories
        if not path.is_dir()
    ]

    print(f"Project: {PROJECT_NAME}")
    print(f"Title: {PROJECT_TITLE}")
    if missing_directories:
        print("Initialization status: incomplete")
        print(f"Missing directories: {', '.join(missing_directories)}")
    else:
        print("Initialization status: ready (foundation only)")

    print("Implemented capabilities:")
    for capability in IMPLEMENTED_CAPABILITIES:
        print(f"- {capability}")
    print("Planned capabilities (not implemented yet):")
    for capability in PLANNED_CAPABILITIES:
        print(f"- {capability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
