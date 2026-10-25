"""Local CPU object detection with mandatory model-integrity verification."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from src.anomaly.checks import (
    AnomalyThresholds,
    ImageAnomalyResult,
    analyze_image,
)
from src.model.integrity import (
    ModelInferenceBlocked,
    ModelVerificationResult,
    ensure_model_verified,
    verify_model,
)
from src.integrity.data_checks import sha256_file


@dataclass(frozen=True)
class Detection:
    """One detected object with class, confidence, and pixel box coordinates."""

    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class InferenceRunResult:
    """Integrity, anomaly, and object-detection outcome for one input image."""

    status: str
    model_verification: ModelVerificationResult
    image_analysis: ImageAnomalyResult
    detections: list[Detection] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    inference_succeeded: bool = False
    confidence: float | None = None


ModelLoader = Callable[[Path], Any]


def _load_yolo_model(model_path: Path) -> Any:
    """Import and construct Ultralytics YOLO lazily after integrity verification."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "YOLO inference requires Ultralytics. Install project dependencies with "
            "'py -3.12 -m pip install -r requirements.txt'."
        ) from error
    return YOLO(str(model_path))


def _extract_detections(prediction_results: Any) -> list[Detection]:
    detections: list[Detection] = []
    for prediction in prediction_results:
        boxes = prediction.boxes
        if boxes is None:
            continue
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            class_name = str(prediction.names.get(class_id, class_id))
            coordinates = boxes.xyxy[index].tolist()
            detections.append(
                Detection(
                    class_id,
                    class_name,
                    float(boxes.conf[index].item()),
                    tuple(float(value) for value in coordinates),
                )
            )
    return detections


def run_inference(
    image_path: Path,
    model_id: str,
    registry_path: Path,
    *,
    confidence: float = 0.25,
    anomaly_thresholds: AnomalyThresholds | None = None,
    model_loader: ModelLoader | None = None,
) -> InferenceRunResult:
    """Analyze an image, verify the exact registered model, then run local CPU detection."""
    if not 0 <= confidence <= 1:
        raise ValueError("Inference confidence must be between 0 and 1.")

    image_analysis = analyze_image(image_path, anomaly_thresholds)
    if not image_analysis.valid:
        review = ModelVerificationResult(
            model_id,
            None,
            None,
            "REVIEW REQUIRED",
            "Inference was not run because the input image is invalid.",
        )
        return InferenceRunResult(
            "REVIEW REQUIRED",
            review,
            image_analysis,
            reasons=image_analysis.reasons,
            confidence=confidence,
        )

    verification = verify_model(model_id, registry_path)
    try:
        ensure_model_verified(verification)
    except ModelInferenceBlocked as error:
        return InferenceRunResult(
            verification.status,
            verification,
            image_analysis,
            reasons=[str(error), *image_analysis.reasons],
            confidence=confidence,
        )

    if verification.file_path is None:
        return InferenceRunResult(
            "REVIEW REQUIRED",
            verification,
            image_analysis,
            reasons=["Verified model registration has no model file path."],
            confidence=confidence,
        )

    registered_path = Path(verification.file_path)
    try:
        with tempfile.TemporaryDirectory(prefix="cv-integrity-model-") as snapshot_dir:
            snapshot_path = Path(snapshot_dir) / registered_path.name
            shutil.copyfile(registered_path, snapshot_path)
            snapshot_digest = sha256_file(snapshot_path)
            if snapshot_digest != verification.expected_hash:
                changed = replace(
                    verification,
                    actual_hash=snapshot_digest,
                    status="SUSPICIOUS",
                    reason=(
                        "CRITICAL: model bytes changed while creating the private "
                        "inference snapshot; loading was blocked."
                    ),
                )
                return InferenceRunResult(
                    "SUSPICIOUS",
                    changed,
                    image_analysis,
                    reasons=[changed.reason, *image_analysis.reasons],
                    confidence=confidence,
                )

            snapshot_verification = replace(
                verification,
                actual_hash=snapshot_digest,
                reason="Private inference snapshot matches the registered baseline.",
            )
            ensure_model_verified(snapshot_verification)
            preload_digest = sha256_file(snapshot_path)
            if preload_digest != verification.expected_hash:
                changed = replace(
                    snapshot_verification,
                    actual_hash=preload_digest,
                    status="SUSPICIOUS",
                    reason=(
                        "CRITICAL: private model snapshot changed before loading; "
                        "inference was blocked."
                    ),
                )
                return InferenceRunResult(
                    "SUSPICIOUS",
                    changed,
                    image_analysis,
                    reasons=[changed.reason, *image_analysis.reasons],
                    confidence=confidence,
                )
            snapshot_verification = replace(
                snapshot_verification,
                actual_hash=preload_digest,
                reason="Private inference snapshot matches the baseline before loading.",
            )
            ensure_model_verified(snapshot_verification)
            loader = model_loader or _load_yolo_model
            model = loader(snapshot_path)
            predictions = model.predict(
                source=str(image_path),
                device="cpu",
                conf=confidence,
                verbose=False,
            )
            detections = _extract_detections(predictions)
    except Exception as error:
        return InferenceRunResult(
            "REVIEW REQUIRED",
            snapshot_verification,
            image_analysis,
            reasons=[f"Inference could not be completed: {error}", *image_analysis.reasons],
            confidence=confidence,
        )

    status = "REVIEW REQUIRED" if image_analysis.reasons else "VERIFIED"
    reasons = list(image_analysis.reasons)
    if not detections:
        reasons.append("No objects were detected.")
    return InferenceRunResult(
        status,
        verification,
        image_analysis,
        detections,
        reasons,
        inference_succeeded=True,
        confidence=confidence,
    )


def format_inference_result(result: InferenceRunResult) -> str:
    """Format image-quality and object-detection results for terminal output."""
    image = result.image_analysis
    lines = [
        f"Run status: {result.status}",
        f"Model verification: {result.model_verification.status}",
        f"Image: {image.image_path}",
    ]
    if image.width is not None and image.height is not None:
        lines.append(f"Dimensions: {image.width}x{image.height}")
        lines.append(f"Aspect ratio: {image.aspect_ratio:.3f}")
        lines.append(f"Mean brightness: {image.mean_brightness:.1f}/255")
        lines.append(f"Laplacian variance: {image.laplacian_variance:.2f}")
    lines.append(f"Detections: {len(result.detections)}")
    for detection in result.detections:
        x1, y1, x2, y2 = detection.box_xyxy
        lines.append(
            f"- {detection.class_name} (class {detection.class_id}), "
            f"confidence {detection.confidence:.3f}, "
            f"box [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )
    lines.append(
        "Confidence is a model score, not proof that a prediction is correct."
    )
    if result.reasons:
        lines.append("Reasons / warnings:")
        lines.extend(f"- {reason}" for reason in result.reasons)
    return "\n".join(lines)
