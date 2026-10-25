"""Local model baseline registration and integrity verification."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.integrity.data_checks import sha256_file


MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class ModelRegistryError(ValueError):
    """Raised when a model registry is missing, malformed, or cannot be saved."""


class ModelInferenceBlocked(RuntimeError):
    """Raised when a model is not verified and must not be used for inference."""


@dataclass(frozen=True)
class ModelVerificationResult:
    """Outcome of comparing a model file with its registered baseline."""

    model_id: str
    expected_hash: str | None
    actual_hash: str | None
    status: str
    reason: str
    file_path: str | None = None


def _validate_model_id(model_id: str) -> None:
    if not MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError(
            "Model ID must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, or hyphens."
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON object-pairs hook that rejects duplicate keys rather than hiding them."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelRegistryError(f"Duplicate key in model registry: {key!r}.")
        result[key] = value
    return result


def _load_registry(registry_path: Path) -> dict[str, dict[str, str]]:
    if not registry_path.exists():
        return {}
    try:
        with registry_path.open("r", encoding="utf-8") as registry_file:
            data = json.load(
                registry_file, object_pairs_hook=_reject_duplicate_json_keys
            )
    except ModelRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelRegistryError(
            f"Unable to read model registry {registry_path}: {error}"
        ) from error

    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ModelRegistryError(
            f"Model registry {registry_path} has an unsupported or invalid format."
        )
    models = data.get("models")
    if not isinstance(models, dict):
        raise ModelRegistryError(
            f"Model registry {registry_path} must contain a models object."
        )

    validated: dict[str, dict[str, str]] = {}
    for model_id, entry in models.items():
        if not isinstance(model_id, str):
            raise ModelRegistryError("Model registry contains a non-text model ID.")
        try:
            _validate_model_id(model_id)
        except ValueError as error:
            raise ModelRegistryError(str(error)) from error
        if not isinstance(entry, dict):
            raise ModelRegistryError(
                f"Registration for {model_id!r} must be a JSON object."
            )
        file_path = entry.get("file_path")
        expected_hash = entry.get("expected_hash")
        registered_at = entry.get("registered_at")
        if (
            entry.get("model_id") != model_id
            or not isinstance(file_path, str)
            or not file_path
            or not isinstance(expected_hash, str)
            or not SHA256_PATTERN.fullmatch(expected_hash)
            or not isinstance(registered_at, str)
        ):
            raise ModelRegistryError(
                f"Registration for {model_id!r} has invalid fields."
            )
        try:
            datetime.fromisoformat(registered_at)
        except ValueError as error:
            raise ModelRegistryError(
                f"Registration timestamp for {model_id!r} is invalid."
            ) from error
        validated[model_id] = {
            "model_id": model_id,
            "file_path": file_path,
            "expected_hash": expected_hash,
            "registered_at": registered_at,
        }
    return validated


def _write_registry(
    registry_path: Path,
    models: dict[str, dict[str, str]],
) -> None:
    """Atomically replace the registry after writing the complete JSON document."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": 1, "models": models}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=registry_path.parent,
            prefix=f".{registry_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(document, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, registry_path)
    except OSError as error:
        raise ModelRegistryError(
            f"Unable to save model registry {registry_path}: {error}"
        ) from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def register_model(
    model_id: str,
    model_path: Path,
    registry_path: Path,
    *,
    replace: bool = False,
) -> dict[str, str]:
    """Record the SHA-256 baseline for a readable model file."""
    _validate_model_id(model_id)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist or is not a file: {model_path}")
    expected_hash = sha256_file(model_path)
    models = _load_registry(registry_path)
    if model_id in models and not replace:
        raise ModelRegistryError(
            f"Model ID {model_id!r} is already registered. "
            "Use --replace to explicitly update its trusted baseline."
        )

    registration = {
        "model_id": model_id,
        "file_path": str(model_path.resolve()),
        "expected_hash": expected_hash,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    models[model_id] = registration
    _write_registry(registry_path, models)
    return registration


def load_model_registry_path(config_path: Path, project_root: Path) -> Path:
    """Load the configured paths.model_registry value without requiring a YAML package."""
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ModelRegistryError(
            f"Unable to read configuration file {config_path}: {error}"
        ) from error

    in_paths = False
    registry_value: str | None = None
    for line in lines:
        content = line.split("#", maxsplit=1)[0].rstrip()
        if not content.strip():
            continue
        if not content[0].isspace():
            in_paths = content.strip() == "paths:"
        elif in_paths:
            key, separator, value = content.strip().partition(":")
            if separator and key == "model_registry":
                registry_value = value.strip().strip("'\"")
                break
    if not registry_value:
        raise ModelRegistryError(
            f"Configuration file {config_path} must define paths.model_registry."
        )
    registry_path = Path(registry_value)
    return registry_path if registry_path.is_absolute() else project_root / registry_path


def verify_model(model_id: str, registry_path: Path) -> ModelVerificationResult:
    """Compare the registered baseline with the current bytes at its file path."""
    try:
        _validate_model_id(model_id)
    except ValueError as error:
        return ModelVerificationResult(
            model_id, None, None, "REVIEW REQUIRED", str(error)
        )

    try:
        models = _load_registry(registry_path)
    except ModelRegistryError as error:
        return ModelVerificationResult(
            model_id, None, None, "REVIEW REQUIRED", str(error)
        )
    registration = models.get(model_id)
    if registration is None:
        return ModelVerificationResult(
            model_id,
            None,
            None,
            "REVIEW REQUIRED",
            "No trusted registration exists for this model ID.",
        )

    model_path = Path(registration["file_path"])
    expected_hash = registration["expected_hash"]
    try:
        actual_hash = sha256_file(model_path)
    except OSError as error:
        return ModelVerificationResult(
            model_id,
            expected_hash,
            None,
            "REVIEW REQUIRED",
            f"Model file is missing or cannot be read: {error}",
            str(model_path),
        )

    if actual_hash == expected_hash:
        return ModelVerificationResult(
            model_id,
            expected_hash,
            actual_hash,
            "VERIFIED",
            "Current file hash matches the registered baseline.",
            str(model_path),
        )
    return ModelVerificationResult(
        model_id,
        expected_hash,
        actual_hash,
        "SUSPICIOUS",
        "CRITICAL: current file hash differs from the registered baseline.",
        str(model_path),
    )


def ensure_model_verified(result: ModelVerificationResult) -> None:
    """Block model use unless verification matched the registered baseline."""
    if result.status != "VERIFIED":
        raise ModelInferenceBlocked(
            f"Model {result.model_id!r} cannot be used: "
            f"{result.status} — {result.reason}"
        )


def format_verification(result: ModelVerificationResult) -> str:
    """Render a verification outcome with hashes when available."""
    lines = [
        f"Model ID: {result.model_id}",
        f"Status: {result.status}",
        f"Reason: {result.reason}",
    ]
    if result.file_path:
        lines.append(f"File: {result.file_path}")
    lines.append(f"Expected SHA-256: {result.expected_hash or 'unavailable'}")
    lines.append(f"Actual SHA-256: {result.actual_hash or 'unavailable'}")
    lines.append(
        "Note: a matching baseline confirms unchanged bytes only; it does not prove "
        "the model's original source, authenticity, accuracy, or safety."
    )
    return "\n".join(lines)
