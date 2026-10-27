"""Local trusted SHA-256 baseline registration for input images."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.integrity.data_checks import sha256_file


SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class ImageRegistryError(ValueError):
    """Raised when an image baseline registry is malformed or cannot be updated."""


@dataclass(frozen=True)
class ImageVerificationResult:
    """Outcome of comparing one image with its trusted local baseline."""

    image_path: str
    expected_hash: str | None
    actual_hash: str | None
    status: str
    reason: str


def _image_key(image_path: Path) -> str:
    """Use a resolved, case-normalized path so a changed file retains its identity."""
    return str(image_path.resolve()).casefold()


def _load_registry(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {"schema_version": 1, "images": {}}
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ImageRegistryError(
            f"Unable to read image baseline registry {registry_path}: {error}"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("images"), dict)
    ):
        raise ImageRegistryError(
            f"Image baseline registry {registry_path} must contain schema_version 1 and images."
        )
    return payload


def register_image(
    image_path: Path,
    registry_path: Path,
    *,
    replace: bool = False,
) -> dict[str, str]:
    """Explicitly record the current bytes of an image as its trusted baseline."""
    resolved = image_path.resolve()
    expected_hash = sha256_file(resolved)
    registry = _load_registry(registry_path)
    key = _image_key(resolved)
    images = registry["images"]
    if key in images and not replace:
        raise ImageRegistryError(
            f"A trusted baseline already exists for {resolved}. "
            "Use --replace to explicitly update it."
        )
    registration = {
        "file_path": str(resolved),
        "expected_hash": expected_hash,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    images[key] = registration
    try:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ImageRegistryError(
            f"Unable to write image baseline registry {registry_path}: {error}"
        ) from error
    return registration


def verify_image(image_path: Path, registry_path: Path) -> ImageVerificationResult:
    """Compare the current image bytes with an explicitly registered baseline."""
    resolved = image_path.resolve()
    display_path = str(resolved)
    try:
        registry = _load_registry(registry_path)
    except ImageRegistryError as error:
        return ImageVerificationResult(
            display_path, None, None, "REVIEW REQUIRED", str(error)
        )

    registration = registry["images"].get(_image_key(resolved))
    if not isinstance(registration, dict):
        return ImageVerificationResult(
            display_path,
            None,
            None,
            "REVIEW REQUIRED",
            "No trusted image baseline exists for this image path.",
        )
    expected_hash = registration.get("expected_hash")
    if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
        return ImageVerificationResult(
            display_path,
            None,
            None,
            "REVIEW REQUIRED",
            "Trusted image baseline is malformed.",
        )
    try:
        actual_hash = sha256_file(resolved)
    except OSError as error:
        return ImageVerificationResult(
            display_path,
            expected_hash,
            None,
            "REVIEW REQUIRED",
            str(error),
        )
    if actual_hash == expected_hash:
        return ImageVerificationResult(
            display_path,
            expected_hash,
            actual_hash,
            "VERIFIED",
            "Current image hash matches the trusted baseline.",
        )
    return ImageVerificationResult(
        display_path,
        expected_hash,
        actual_hash,
        "SUSPICIOUS",
        "CRITICAL: current image hash differs from the trusted baseline.",
    )


def format_image_verification(result: ImageVerificationResult) -> str:
    """Render image verification evidence for the CLI."""
    return "\n".join(
        (
            f"Image integrity: {result.status}",
            f"Image: {result.image_path}",
            f"Expected SHA-256: {result.expected_hash or 'unavailable'}",
            f"Actual SHA-256: {result.actual_hash or 'unavailable'}",
            f"Reason: {result.reason}",
        )
    )
