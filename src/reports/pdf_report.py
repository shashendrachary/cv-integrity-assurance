"""Local PDF assurance report generation."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo
from pathlib import Path
from xml.sax.saxutils import escape

from src.provenance.records import ProvenanceRecord


class ReportGenerationError(RuntimeError):
    """Raised when an assurance PDF cannot be generated."""


PROJECT_TITLE = (
    "Trustworthy Computer Vision Integrity Assurance for Data, Models "
    "and Inference Outputs in Multi-Contributor Pipelines"
)


@dataclass(frozen=True)
class BatchImageReportItem:
    """Compact integrity and inference evidence for one batch image."""

    filename: str
    image_sha256: str | None
    width: int | None
    height: int | None
    image_format: str | None
    duplicate_status: str
    quality_flags: tuple[str, ...]
    status: str
    model_verification_status: str
    inference_succeeded: bool
    reasons: tuple[str, ...]
    detections: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BatchReportSummary:
    """Outcome totals and selected model information for one batch report."""

    discovered: int
    processed: int
    successful: int
    review_required: int
    failures: int
    inference_failures: int
    model_id: str
    model_verification_status: str


@dataclass(frozen=True)
class VideoFrameReportItem:
    """Compact outcome and detection evidence for one sampled video frame."""

    frame_index: int
    time_seconds: float
    status: str
    detection_count: int
    model_verification_status: str
    findings: tuple[str, ...]
    detections: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class VideoReportSummary:
    """Video metadata and outcome totals for a consolidated report."""

    video_filename: str
    video_path: str
    interval_seconds: float
    model_id: str
    sampled_frames: int
    verified_frames: int
    review_required_frames: int
    failed_frames: int
    model_verification_status: str


def load_report_directory(config_path: Path, project_root: Path) -> Path:
    """Read paths.reports from the simple project YAML configuration."""
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReportGenerationError(
            f"Unable to read report configuration {config_path}: {error}"
        ) from error
    in_paths = False
    report_value: str | None = None
    for line in lines:
        content = line.split("#", maxsplit=1)[0].rstrip()
        if not content.strip():
            continue
        if not content[0].isspace():
            in_paths = content.strip() == "paths:"
        elif in_paths:
            key, separator, value = content.strip().partition(":")
            if separator and key == "reports":
                report_value = value.strip().strip("'\"")
                break
    if not report_value:
        raise ReportGenerationError(
            f"Configuration file {config_path} must define paths.reports."
        )
    directory = Path(report_value)
    return directory if directory.is_absolute() else project_root / directory


def _safe_text(value: object) -> str:
    return escape(str(value)).replace("\n", "<br/>")


def _unique_report_path(report_directory: Path, run_id: str) -> Path:
    """Reserve a unique report filename, avoiding overwrites."""
    safe_id = "".join(
        character for character in run_id if character.isalnum() or character in "-_"
    )
    if not safe_id:
        safe_id = "assurance-run"
    report_directory.mkdir(parents=True, exist_ok=True)
    candidate = report_directory / f"assurance-{safe_id}.pdf"
    suffix = 1
    while True:
        try:
            with candidate.open("xb"):
                return candidate
        except FileExistsError:
            candidate = report_directory / f"assurance-{safe_id}-{suffix}.pdf"
            suffix += 1
        except OSError as error:
            raise ReportGenerationError(
                f"Cannot reserve PDF report path in {report_directory}: {error}"
            ) from error


def generate_pdf_report(
    record: ProvenanceRecord,
    overall_status: str,
    database_status: str,
    report_directory: Path,
) -> Path:
    """Generate a unique, readable PDF containing collected run evidence."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as error:
        raise ReportGenerationError(
            "PDF generation requires ReportLab. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error

    output_path = _unique_report_path(report_directory, record.run_id)
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".assurance-report-", suffix=".tmp", dir=report_directory
        )
    except OSError as error:
        output_path.unlink(missing_ok=True)
        raise ReportGenerationError(
            f"Cannot create temporary PDF report in {report_directory}: {error}"
        ) from error
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            fontSize=16,
            leading=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#17365D"),
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(ParagraphStyle(name="SmallText", parent=styles["BodyText"], fontSize=8))

    def paragraph(text: object, style: str = "BodyText") -> Paragraph:
        return Paragraph(_safe_text(text), styles[style])

    story = [
        paragraph(PROJECT_TITLE, "ReportTitle"),
        Spacer(1, 8),
        paragraph("Assurance Report", "Heading1"),
        paragraph(f"Run ID: {record.run_id}"),
        paragraph(
            "Indian Standard Time: "
            + record.recorded_at.astimezone(
                ZoneInfo("Asia/Kolkata")
            ).strftime("%d %B %Y, %I:%M:%S %p IST")
        ),
        paragraph(f"Overall status: {overall_status}", "Heading2"),
        paragraph(
            "Rule-based finding only. A matching model hash establishes unchanged "
            "bytes since registration; it does not prove source authenticity, "
            "accuracy, or safety."
        ),
    ]

    def add_table(title: str, rows: list[tuple[str, object]]) -> None:
        story.append(paragraph(title, "SectionHeading"))
        table = Table(
            [[paragraph(label, "SmallText"), paragraph(value, "SmallText")] for label, value in rows],
            colWidths=[1.65 * inch, 5.45 * inch],
            repeatRows=0,
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F6")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(table)

    add_table(
        "Input image",
        [
            ("Path", record.image_path),
            ("SHA-256", record.image_sha256 or "Unavailable"),
            ("Dimensions", f"{record.image_metrics.get('width')} x {record.image_metrics.get('height')}"),
            ("Aspect ratio", record.image_metrics.get("aspect_ratio", "Unavailable")),
        ],
    )
    add_table(
        "Model integrity",
        [
            ("Model ID", record.model_id),
            ("Model path", record.model_path or "Unavailable"),
            ("Verification", record.model_verification_status),
            ("Expected SHA-256", record.model_expected_sha256 or "Unavailable"),
            ("Actual SHA-256", record.model_actual_sha256 or "Unavailable"),
        ],
    )
    story.append(paragraph("Inference results", "SectionHeading"))
    story.append(
        paragraph(
            f"Inference succeeded: {record.inference_succeeded}; "
            f"confidence threshold: {record.confidence if record.confidence is not None else 'Unavailable'}"
        )
    )
    if record.detections:
        detection_rows: list[list[object]] = [
            ["Class", "ID", "Confidence", "Bounding box (x1, y1, x2, y2)"]
        ]
        detection_rows.extend(
            [
                str(item["class_name"]),
                str(item["class_id"]),
                f"{float(item['confidence']):.3f}",
                str(item["box_xyxy"]),
            ]
            for item in record.detections
        )
        table = Table(
            [[paragraph(value, "SmallText") for value in row] for row in detection_rows],
            colWidths=[1.55 * inch, 0.55 * inch, 1.0 * inch, 4.0 * inch],
            repeatRows=1,
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E6F2")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)
    else:
        story.append(paragraph("No detections were available for this run."))

    add_table(
        "Image-quality measurements and configuration",
        [
            ("Measurements", record.image_metrics),
            ("Thresholds / configuration", record.configuration),
        ],
    )
    add_table(
        "Evidence persistence",
        [
            ("PostgreSQL status", database_status),
            ("Provenance run ID", record.run_id),
        ],
    )
    story.append(paragraph("Reasons and warnings", "SectionHeading"))
    if record.reasons:
        story.extend(paragraph(f"• {reason}") for reason in record.reasons)
    else:
        story.append(paragraph("No review reasons were recorded."))
    story.append(paragraph("Limitations", "SectionHeading"))
    story.append(
        paragraph(
            "Image anomaly checks use initial heuristics that require dataset-specific "
            "calibration. Detection confidence is not correctness proof. Provenance "
            "records describe evidence observed by this application and are not "
            "independent proof of model authenticity or prediction validity."
        )
    )

    try:
        document = SimpleDocTemplate(
            str(temporary_path),
            pagesize=letter,
            rightMargin=0.7 * inch,
            leftMargin=0.7 * inch,
            topMargin=0.65 * inch,
            bottomMargin=0.65 * inch,
            title=f"Assurance Report {record.run_id}",
            author="CV-Integrity-Assurance",
        )
        document.build(story)
        os.replace(temporary_path, output_path)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise OSError("generated PDF is missing or empty")
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise ReportGenerationError(f"Could not generate PDF report: {error}") from error
    return output_path


def generate_batch_pdf_report(
    items: list[BatchImageReportItem],
    summary: BatchReportSummary,
    report_directory: Path,
) -> tuple[Path, Path]:
    """Generate one consolidated PDF and a JSON file with detection details."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            LongTable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as error:
        raise ReportGenerationError(
            "PDF generation requires ReportLab. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error

    report_id = f"batch-{uuid.uuid4().hex}"
    report_path = _unique_report_path(report_directory, report_id)
    sidecar_path = report_path.with_suffix(".json")
    temporary_path: Path | None = None
    sidecar_created = False
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BatchReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            fontSize=16,
            leading=20,
        )
    )
    styles.add(ParagraphStyle(name="BatchSmallText", parent=styles["BodyText"], fontSize=7))

    def paragraph(value: object, style: str = "BodyText") -> Paragraph:
        return Paragraph(_safe_text(value), styles[style])

    generated_at = datetime.now(timezone.utc)
    story = [
        paragraph(PROJECT_TITLE, "BatchReportTitle"),
        Spacer(1, 8),
        paragraph("Consolidated Batch Assurance Report", "Heading1"),
        paragraph(
            "Generated at: "
            + generated_at.astimezone(ZoneInfo("Asia/Kolkata")).strftime(
                "%d %B %Y, %I:%M:%S %p IST"
            )
        ),
    ]

    story.append(paragraph("Executive Summary", "Heading2"))
    summary_rows = [
        ["Images discovered", str(summary.discovered)],
        ["Images processed", str(summary.processed)],
        ["Successful assessments", str(summary.successful)],
        ["Review-required images", str(summary.review_required)],
        ["Failures", str(summary.failures)],
        ["Selected model ID", summary.model_id],
        ["Model verification status", summary.model_verification_status],
        ["Report generation timestamp", generated_at.isoformat()],
    ]
    story.append(
        Table(
            [[paragraph(key), paragraph(value)] for key, value in summary_rows],
            colWidths=[2.5 * inch, 7.0 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F6")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            ),
        )
    )

    story.append(paragraph("Image Integrity and Properties", "Heading2"))
    image_rows: list[list[object]] = [
        [
            "Filename",
            "SHA-256",
            "Dimensions",
            "Format",
            "Duplicate",
            "Quality / integrity flags",
            "Status",
        ]
    ]
    for item in items:
        dimensions = (
            f"{item.width} x {item.height}"
            if item.width is not None and item.height is not None
            else "N/A"
        )
        image_rows.append(
            [
                paragraph(item.filename, "BatchSmallText"),
                paragraph(item.image_sha256 or "N/A", "BatchSmallText"),
                paragraph(dimensions, "BatchSmallText"),
                paragraph(item.image_format or "N/A", "BatchSmallText"),
                paragraph(item.duplicate_status, "BatchSmallText"),
                paragraph("; ".join(item.quality_flags) or "None", "BatchSmallText"),
                paragraph(item.status, "BatchSmallText"),
            ]
        )
    image_table = LongTable(
        image_rows,
        colWidths=[
            1.35 * inch,
            2.0 * inch,
            0.8 * inch,
            0.65 * inch,
            0.75 * inch,
            2.5 * inch,
            0.9 * inch,
        ],
        repeatRows=1,
        splitByRow=1,
    )
    image_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E6F2")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(image_table)

    detection_count = sum(len(item.detections) for item in items)
    class_counts: dict[str, int] = {}
    for item in items:
        for detection in item.detections:
            class_name = str(detection.get("class_name", "N/A"))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    story.append(paragraph("AI Results Summary", "Heading2"))
    story.append(
        paragraph(
            f"Aggregate detection count: {detection_count}; model ID: "
            f"{summary.model_id}; model verification: "
            f"{summary.model_verification_status}; inference failures: "
            f"{summary.inference_failures}."
        )
    )
    story.append(
        paragraph(
            "Object-class counts: "
            + (
                ", ".join(
                    f"{class_name}: {count}"
                    for class_name, count in sorted(class_counts.items())
                )
                if class_counts
                else "N/A"
            )
        )
    )

    story.append(paragraph("Findings and Recommendations", "Heading2"))
    flagged_items = [item for item in items if item.status != "VERIFIED"]
    if not flagged_items:
        story.append(paragraph("No images were flagged for review or failure."))
    else:
        story.append(
            paragraph(
                "Review flagged images individually, check the concise reasons below, "
                "and resolve image-quality or integrity issues before relying on results."
            )
        )
        for item in flagged_items:
            reasons = "; ".join(item.reasons) or "No additional reason was recorded."
            story.append(paragraph(f"{item.filename} ({item.status}): {reasons}"))

    sidecar_data = {
        "report_id": report_path.stem,
        "summary": {
            "discovered": summary.discovered,
            "processed": summary.processed,
            "successful": summary.successful,
            "review_required": summary.review_required,
            "failures": summary.failures,
            "inference_failures": summary.inference_failures,
            "model_id": summary.model_id,
            "model_verification_status": summary.model_verification_status,
            "generated_at": generated_at.isoformat(),
        },
        "ai_results": {
            "detection_count": detection_count,
            "object_class_counts": class_counts,
        },
        "images": [
            {
                "filename": item.filename,
                "status": item.status,
                "image_sha256": item.image_sha256,
                "detections": list(item.detections),
            }
            for item in items
        ],
    }

    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".assurance-batch-", suffix=".tmp", dir=report_directory
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        document = SimpleDocTemplate(
            str(temporary_path),
            pagesize=landscape(letter),
            rightMargin=0.45 * inch,
            leftMargin=0.45 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            title=f"Consolidated Batch Assurance Report {report_path.stem}",
            author="CV-Integrity-Assurance",
        )
        document.build(story)
        if sidecar_path.exists():
            raise FileExistsError(f"Detection sidecar already exists: {sidecar_path}")
        with sidecar_path.open("x", encoding="utf-8") as sidecar_file:
            sidecar_created = True
            json.dump(sidecar_data, sidecar_file, indent=2, ensure_ascii=False)
            sidecar_file.write("\n")
        os.replace(temporary_path, report_path)
        temporary_path = None
        if not report_path.is_file() or report_path.stat().st_size == 0:
            raise OSError("generated batch PDF is missing or empty")
    except Exception as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        if sidecar_created:
            sidecar_path.unlink(missing_ok=True)
        raise ReportGenerationError(
            f"Could not generate consolidated batch report: {error}"
        ) from error
    return report_path, sidecar_path


def generate_video_pdf_report(
    frames: list[VideoFrameReportItem],
    summary: VideoReportSummary,
    report_directory: Path,
) -> tuple[Path, Path]:
    """Generate one consolidated video PDF and a matching frame-results JSON."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            LongTable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as error:
        raise ReportGenerationError(
            "PDF generation requires ReportLab. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error

    report_id = f"video-{uuid.uuid4().hex}"
    report_path = _unique_report_path(report_directory, report_id)
    sidecar_path = report_path.with_suffix(".json")
    temporary_path: Path | None = None
    sidecar_created = False

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="VideoReportTitle",
            parent=styles["Title"],
            alignment=1,
            textColor=colors.HexColor("#17365D"),
            fontSize=16,
            leading=20,
        )
    )
    styles.add(
        ParagraphStyle(name="VideoSmallText", parent=styles["BodyText"], fontSize=8)
    )

    def paragraph(value: object, style: str = "BodyText") -> Paragraph:
        return Paragraph(_safe_text(value), styles[style])

    generated_at = datetime.now(timezone.utc)
    story = [
        paragraph(PROJECT_TITLE, "VideoReportTitle"),
        Spacer(1, 8),
        paragraph("Consolidated Video Assurance Report", "Heading1"),
        paragraph(f"Video filename: {summary.video_filename}"),
        paragraph(f"Video path: {summary.video_path}"),
        paragraph(f"Sampling interval: {summary.interval_seconds:g} seconds"),
        paragraph(f"Selected model ID: {summary.model_id}"),
        paragraph(
            "Report timestamp: "
            + generated_at.astimezone(ZoneInfo("Asia/Kolkata")).strftime(
                "%d %B %Y, %I:%M:%S %p IST"
            )
        ),
    ]

    story.append(paragraph("Executive Summary", "Heading2"))
    totals = [
        ["Total sampled frames", str(summary.sampled_frames)],
        ["Verified frames", str(summary.verified_frames)],
        ["Review-required frames", str(summary.review_required_frames)],
        ["Failed frames", str(summary.failed_frames)],
        ["Model verification status", summary.model_verification_status],
    ]
    story.append(
        Table(
            [[paragraph(key), paragraph(value)] for key, value in totals],
            colWidths=[2.5 * inch, 7.0 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F6")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            ),
        )
    )

    story.append(paragraph("Sampled Frame Results", "Heading2"))
    table_rows: list[list[object]] = [
        ["Frame index", "Time (seconds)", "Status", "Detections", "Concise findings"]
    ]
    for frame in frames:
        table_rows.append(
            [
                paragraph(frame.frame_index, "VideoSmallText"),
                paragraph(f"{frame.time_seconds:.3f}", "VideoSmallText"),
                paragraph(frame.status, "VideoSmallText"),
                paragraph(frame.detection_count, "VideoSmallText"),
                paragraph("; ".join(frame.findings) or "None", "VideoSmallText"),
            ]
        )
    result_table = LongTable(
        table_rows,
        colWidths=[0.8 * inch, 1.15 * inch, 1.2 * inch, 0.8 * inch, 6.05 * inch],
        repeatRows=1,
        splitByRow=1,
    )
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E6F2")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(result_table)

    class_counts: dict[str, int] = {}
    for frame in frames:
        for detection in frame.detections:
            class_name = str(detection.get("class_name", "N/A"))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    detection_total = sum(frame.detection_count for frame in frames)
    story.append(paragraph("AI Results Summary", "Heading2"))
    story.append(
        paragraph(
            f"Total detections: {detection_total}. Object-class counts: "
            + (
                ", ".join(
                    f"{class_name}: {count}"
                    for class_name, count in sorted(class_counts.items())
                )
                if class_counts
                else "N/A"
            )
        )
    )

    story.append(paragraph("Findings", "Heading2"))
    flagged_frames = [
        frame for frame in frames if frame.status != "VERIFIED"
    ]
    if not flagged_frames:
        story.append(paragraph("No sampled frames required review or failed."))
    else:
        for frame in flagged_frames:
            findings = "; ".join(frame.findings) or "No additional reason was recorded."
            story.append(
                paragraph(
                    f"Frame {frame.frame_index} at {frame.time_seconds:.3f}s "
                    f"({frame.status}): {findings}"
                )
            )

    sidecar_data = {
        "report_id": report_path.stem,
        "video": {
            "filename": summary.video_filename,
            "path": summary.video_path,
            "sampling_interval_seconds": summary.interval_seconds,
        },
        "summary": {
            "sampled_frames": summary.sampled_frames,
            "verified_frames": summary.verified_frames,
            "review_required_frames": summary.review_required_frames,
            "failed_frames": summary.failed_frames,
            "model_id": summary.model_id,
            "model_verification_status": summary.model_verification_status,
            "generated_at": generated_at.isoformat(),
        },
        "ai_results": {
            "detection_count": detection_total,
            "object_class_counts": class_counts,
        },
        "frames": [
            {
                "frame_index": frame.frame_index,
                "time_seconds": frame.time_seconds,
                "status": frame.status,
                "detection_count": frame.detection_count,
                "model_verification_status": frame.model_verification_status,
                "findings": list(frame.findings),
                "detections": list(frame.detections),
            }
            for frame in frames
        ],
    }

    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".assurance-video-report-", suffix=".tmp", dir=report_directory
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        document = SimpleDocTemplate(
            str(temporary_path),
            pagesize=landscape(letter),
            rightMargin=0.45 * inch,
            leftMargin=0.45 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            title=f"Video Assurance Report {report_path.stem}",
            author="CV-Integrity-Assurance",
        )
        document.build(story)
        with sidecar_path.open("x", encoding="utf-8") as sidecar_file:
            sidecar_created = True
            json.dump(sidecar_data, sidecar_file, indent=2, ensure_ascii=False)
            sidecar_file.write("\n")
        os.replace(temporary_path, report_path)
        temporary_path = None
        if not report_path.is_file() or report_path.stat().st_size == 0:
            raise OSError("generated video PDF is missing or empty")
    except Exception as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        if sidecar_created:
            sidecar_path.unlink(missing_ok=True)
        raise ReportGenerationError(
            f"Could not generate consolidated video report: {error}"
        ) from error
    return report_path, sidecar_path
