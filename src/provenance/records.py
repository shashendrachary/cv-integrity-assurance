"""Inference evidence conversion and parameterized PostgreSQL persistence."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.database.postgres import (
    ConnectionFactory,
    ConnectionSettings,
    run_statement,
)
from src.integrity.data_checks import sha256_file
from src.vision.inference import InferenceRunResult


@dataclass(frozen=True)
class ProvenanceRecord:
    """Serializable evidence actually obtained from one inference attempt."""

    run_id: str
    recorded_at: datetime
    image_path: str
    image_sha256: str | None
    model_id: str
    model_path: str | None
    model_expected_sha256: str | None
    model_actual_sha256: str | None
    model_verification_status: str
    confidence: float | None
    detections: list[dict[str, Any]]
    image_metrics: dict[str, Any]
    configuration: dict[str, Any]
    run_status: str
    inference_succeeded: bool
    reasons: list[str]


def create_provenance_record(
    image_path: Path,
    result: InferenceRunResult,
    configuration: dict[str, Any],
) -> ProvenanceRecord:
    """Build a provenance record, leaving unavailable evidence as null."""
    try:
        image_hash = sha256_file(image_path)
    except OSError:
        image_hash = None

    analysis = result.image_analysis
    metrics: dict[str, Any] = {
        "valid": analysis.valid,
        "width": analysis.width,
        "height": analysis.height,
        "aspect_ratio": analysis.aspect_ratio,
        "mean_brightness": analysis.mean_brightness,
        "laplacian_variance": analysis.laplacian_variance,
    }
    detections = [
        {
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "box_xyxy": list(detection.box_xyxy),
        }
        for detection in result.detections
    ]
    return ProvenanceRecord(
        run_id=str(uuid.uuid4()),
        recorded_at=datetime.now(timezone.utc),
        image_path=str(image_path),
        image_sha256=image_hash,
        model_id=result.model_verification.model_id,
        model_path=result.model_verification.file_path,
        model_expected_sha256=result.model_verification.expected_hash,
        model_actual_sha256=result.model_verification.actual_hash,
        model_verification_status=result.model_verification.status,
        confidence=result.confidence,
        detections=detections,
        image_metrics=metrics,
        configuration=configuration,
        run_status=result.status,
        inference_succeeded=result.inference_succeeded,
        reasons=list(result.reasons),
    )


def insert_provenance_record(
    record: ProvenanceRecord,
    settings: ConnectionSettings,
    connection_factory: ConnectionFactory | None = None,
) -> None:
    """Insert one record using parameters and JSONB casts inside a transaction."""
    statement = """
        INSERT INTO inference_provenance (
            run_id, recorded_at, image_path, image_sha256, model_id, model_path,
            model_expected_sha256, model_actual_sha256, model_verification_status,
            confidence, detections, image_metrics, configuration, run_status,
            inference_succeeded, reasons
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb
        )
    """
    parameters = (
        record.run_id,
        record.recorded_at,
        record.image_path,
        record.image_sha256,
        record.model_id,
        record.model_path,
        record.model_expected_sha256,
        record.model_actual_sha256,
        record.model_verification_status,
        record.confidence,
        json.dumps(record.detections),
        json.dumps(record.image_metrics),
        json.dumps(record.configuration),
        record.run_status,
        record.inference_succeeded,
        json.dumps(record.reasons),
    )
    run_statement(
        settings,
        statement,
        parameters,
        connection_factory=connection_factory,
    )
