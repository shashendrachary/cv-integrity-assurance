"""Local PDF assurance report generation."""

from __future__ import annotations

import os
import tempfile
from datetime import timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

from src.provenance.records import ProvenanceRecord


class ReportGenerationError(RuntimeError):
    """Raised when an assurance PDF cannot be generated."""


PROJECT_TITLE = (
    "Trustworthy Computer Vision Integrity Assurance for Data, Models "
    "and Inference Outputs in Multi-Contributor Pipelines"
)


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
