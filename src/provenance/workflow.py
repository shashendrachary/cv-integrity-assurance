"""End-to-end assurance assessment orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from src.anomaly.checks import AnomalyThresholds
from src.database.postgres import (
    ConnectionSettings,
    DatabaseOperationError,
)
from src.provenance.records import (
    ProvenanceRecord,
    create_provenance_record,
    insert_provenance_record,
)
from src.reports.pdf_report import generate_pdf_report
from src.project_context import ProjectContext
from src.vision.inference import InferenceRunResult, run_inference


InferenceFunction = Callable[..., InferenceRunResult]
RecordInserter = Callable[..., None]
ReportGenerator = Callable[..., Path]


@dataclass(frozen=True)
class AssessmentResult:
    """Complete local assessment, persistence outcome, and generated PDF path."""

    inference: InferenceRunResult
    provenance: ProvenanceRecord
    overall_status: str
    persistence_status: str
    report_path: Path | None


def classify_assessment(
    inference: InferenceRunResult,
) -> tuple[str, list[str]]:
    """Apply priority rules: integrity mismatches, then incomplete/review evidence."""
    reasons = list(inference.reasons)
    if (
        inference.model_verification.status == "SUSPICIOUS"
        or inference.image_verification.status == "SUSPICIOUS"
    ):
        return "SUSPICIOUS", reasons

    review_required = False
    if inference.model_verification.status != "VERIFIED":
        reasons.append("Model integrity verification is incomplete.")
        review_required = True
    if inference.image_verification.status != "VERIFIED":
        reasons.append("Image integrity verification is incomplete.")
        review_required = True
    if not inference.image_analysis.valid:
        reasons.append("Input image validation did not pass.")
        review_required = True
    if inference.image_analysis.reasons:
        reasons.extend(
            reason for reason in inference.image_analysis.reasons if reason not in reasons
        )
        review_required = True
    if inference.status == "REVIEW REQUIRED":
        review_required = True
    if not inference.inference_succeeded:
        reasons.append("Inference did not complete successfully.")
        review_required = True

    if review_required:
        return "REVIEW REQUIRED", reasons
    return "VERIFIED", reasons


def run_assessment(
    image_path: Path,
    model_id: str,
    registry_path: Path,
    report_directory: Path,
    *,
    confidence: float,
    anomaly_thresholds: AnomalyThresholds,
    database_settings: ConnectionSettings | None,
    database_note: str | None = None,
    model_loader: Any = None,
    project_root: Path | None = None,
    image_registry_path: Path | None = None,
    project_context: ProjectContext | None = None,
    provenance_image_path: Path | None = None,
    configuration_metadata: dict[str, Any] | None = None,
    generate_report: bool = True,
    inference_function: InferenceFunction = run_inference,
    record_inserter: RecordInserter = insert_provenance_record,
    report_generator: ReportGenerator = generate_pdf_report,
) -> AssessmentResult:
    """Run inference once, persist its evidence when available, then generate its PDF."""
    inference_options: dict[str, Any] = {
        "confidence": confidence,
        "anomaly_thresholds": anomaly_thresholds,
        "model_loader": model_loader,
    }
    if project_root is not None:
        inference_options["project_root"] = project_root
    if image_registry_path is not None:
        inference_options["image_registry_path"] = image_registry_path
    inference = inference_function(
        image_path,
        model_id,
        registry_path,
        **inference_options,
    )

    configuration = {
        "confidence": confidence,
        "device": "cpu",
        "image_integrity": {
            "status": inference.image_verification.status,
            "expected_sha256": inference.image_verification.expected_hash,
            "actual_sha256": inference.image_verification.actual_hash,
            "reason": inference.image_verification.reason,
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
    }
    if configuration_metadata is not None:
        configuration["assessment_metadata"] = dict(configuration_metadata)

    persistence_status = "not saved"
    if database_settings is None:
        persistence_status = database_note or "skipped (PostgreSQL is not configured)"
    else:
        provisional_status, provisional_reasons = classify_assessment(inference)
        provisional_result = replace(
            inference,
            status=provisional_status,
            reasons=provisional_reasons,
        )
        provisional_record = create_provenance_record(
            image_path,
            provisional_result,
            configuration,
            project_context,
        )
        if provenance_image_path is not None:
            provisional_record = replace(
                provisional_record,
                image_path=str(provenance_image_path),
            )
        try:
            record_inserter(provisional_record, database_settings)
            persistence_status = "saved to PostgreSQL"
        except DatabaseOperationError as error:
            persistence_status = f"not saved: {error}"

    # persistence_succeeded = persistence_status == "saved to PostgreSQL"
    overall_status, reasons = classify_assessment(inference)
    final_inference = replace(inference, status=overall_status, reasons=reasons)

    if database_settings is None:
        record = create_provenance_record(
            image_path,
            final_inference,
            configuration,
            project_context,
        )
        if provenance_image_path is not None:
            record = replace(record, image_path=str(provenance_image_path))
    else:
        record = replace(
            provisional_record,
            run_status=overall_status,
            reasons=reasons,
        )

    report_path = (
        report_generator(
            record,
            overall_status,
            persistence_status,
            report_directory,
        )
        if generate_report
        else None
    )
    return AssessmentResult(
        inference,
        record,
        overall_status,
        persistence_status,
        report_path,
    )
