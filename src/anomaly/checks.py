"""Initial heuristic image-quality checks for inference inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class AnomalyThresholds:
    """Configurable image checks; defaults are initial, uncalibrated heuristics."""

    minimum_width: int = 16
    minimum_height: int = 16
    maximum_width: int = 10000
    maximum_height: int = 10000
    minimum_aspect_ratio: float = 0.1
    maximum_aspect_ratio: float = 10.0
    minimum_mean_brightness: float = 5.0
    maximum_mean_brightness: float = 250.0
    minimum_laplacian_variance: float = 5.0


@dataclass(frozen=True)
class ImageAnomalyResult:
    """Image validity, measured properties, and heuristic review reasons."""

    image_path: str
    valid: bool
    width: int | None = None
    height: int | None = None
    aspect_ratio: float | None = None
    mean_brightness: float | None = None
    laplacian_variance: float | None = None
    reasons: list[str] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _laplacian_variance(pixels: list[int], width: int, height: int) -> float:
    """Calculate a four-neighbour Laplacian variance on a small grayscale image."""
    if width < 3 or height < 3:
        return 0.0
    laplacians: list[float] = []
    for y in range(1, height - 1):
        row_start = y * width
        for x in range(1, width - 1):
            index = row_start + x
            laplacians.append(
                float(
                    pixels[index - width]
                    + pixels[index + width]
                    + pixels[index - 1]
                    + pixels[index + 1]
                    - 4 * pixels[index]
                )
            )
    average = _mean(laplacians)
    return _mean([(value - average) ** 2 for value in laplacians])


def analyze_image(
    image_path: Path,
    thresholds: AnomalyThresholds | None = None,
) -> ImageAnomalyResult:
    """Decode an image and report unusual dimensions, brightness, or blur."""
    active_thresholds = thresholds or AnomalyThresholds()
    try:
        if not image_path.is_file():
            return ImageAnomalyResult(
                str(image_path), False, reasons=[f"Image file is missing: {image_path}"]
            )
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                return ImageAnomalyResult(
                    str(image_path), False, reasons=["Image has invalid dimensions."]
                )
            grayscale = image.convert("L")
            grayscale.thumbnail((256, 256))
            small_width, small_height = grayscale.size
            pixels = list(grayscale.tobytes())
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        return ImageAnomalyResult(
            str(image_path), False, reasons=[f"Image is invalid or unreadable: {error}"]
        )

    aspect_ratio = width / height
    mean_brightness = _mean([float(value) for value in pixels])
    blur_metric = _laplacian_variance(pixels, small_width, small_height)
    reasons: list[str] = []

    if width < active_thresholds.minimum_width or height < active_thresholds.minimum_height:
        reasons.append(
            f"Image dimensions are small ({width}x{height}); minimum is "
            f"{active_thresholds.minimum_width}x{active_thresholds.minimum_height}."
        )
    if width > active_thresholds.maximum_width or height > active_thresholds.maximum_height:
        reasons.append(
            f"Image dimensions are unusually large ({width}x{height})."
        )
    if not active_thresholds.minimum_aspect_ratio <= aspect_ratio <= active_thresholds.maximum_aspect_ratio:
        reasons.append(f"Aspect ratio {aspect_ratio:.3f} is outside configured limits.")
    if mean_brightness < active_thresholds.minimum_mean_brightness:
        reasons.append(f"Mean brightness is low ({mean_brightness:.1f}/255).")
    if mean_brightness > active_thresholds.maximum_mean_brightness:
        reasons.append(f"Mean brightness is high ({mean_brightness:.1f}/255).")
    if blur_metric < active_thresholds.minimum_laplacian_variance:
        reasons.append(
            f"Image may be blurry (Laplacian variance {blur_metric:.2f})."
        )

    return ImageAnomalyResult(
        str(image_path),
        True,
        width,
        height,
        aspect_ratio,
        mean_brightness,
        blur_metric,
        reasons,
    )


def load_inference_settings(config_path: Path) -> tuple[AnomalyThresholds, float]:
    """Read simple scalar inference and anomaly settings from config.yaml."""
    sections: dict[str, dict[str, str]] = {"inference": {}, "anomaly_detection": {}}
    current_section: str | None = None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        content = line.split("#", maxsplit=1)[0].rstrip()
        if not content.strip():
            continue
        if not content[0].isspace():
            name = content.strip().removesuffix(":")
            current_section = name if name in sections else None
        elif current_section:
            key, separator, value = content.strip().partition(":")
            if separator:
                sections[current_section][key] = value.strip().strip("'\"")

    anomaly = sections["anomaly_detection"]
    defaults = AnomalyThresholds()
    try:
        thresholds = AnomalyThresholds(
            minimum_width=int(anomaly.get("minimum_width", defaults.minimum_width)),
            minimum_height=int(anomaly.get("minimum_height", defaults.minimum_height)),
            maximum_width=int(anomaly.get("maximum_width", defaults.maximum_width)),
            maximum_height=int(anomaly.get("maximum_height", defaults.maximum_height)),
            minimum_aspect_ratio=float(
                anomaly.get("minimum_aspect_ratio", defaults.minimum_aspect_ratio)
            ),
            maximum_aspect_ratio=float(
                anomaly.get("maximum_aspect_ratio", defaults.maximum_aspect_ratio)
            ),
            minimum_mean_brightness=float(
                anomaly.get(
                    "minimum_mean_brightness", defaults.minimum_mean_brightness
                )
            ),
            maximum_mean_brightness=float(
                anomaly.get(
                    "maximum_mean_brightness", defaults.maximum_mean_brightness
                )
            ),
            minimum_laplacian_variance=float(
                anomaly.get(
                    "minimum_laplacian_variance",
                    defaults.minimum_laplacian_variance,
                )
            ),
        )
        confidence = float(sections["inference"].get("confidence", "0.25"))
    except ValueError as error:
        raise ValueError(f"Invalid inference or anomaly threshold in {config_path}: {error}") from error
    if not 0 <= confidence <= 1:
        raise ValueError("Inference confidence must be between 0 and 1.")
    return thresholds, confidence
