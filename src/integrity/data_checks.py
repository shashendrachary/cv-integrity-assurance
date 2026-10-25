"""Dataset image and YOLO-label integrity checks."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
HASH_CHUNK_SIZE = 1024 * 1024


class ImageDependencyError(RuntimeError):
    """Raised when Pillow is not installed for image decoding."""


@dataclass
class DataCheckResult:
    """Collected dataset findings and operational scan errors."""

    images_examined: int = 0
    valid_images: int = 0
    invalid_images: list[str] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)
    valid_label_files: int = 0
    invalid_label_files: list[str] = field(default_factory=list)
    empty_label_files: list[str] = field(default_factory=list)
    images_missing_labels: list[str] = field(default_factory=list)
    labels_missing_images: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    operational_errors: list[str] = field(default_factory=list)
    data_available: bool = False


def sha256_file(file_path: Path) -> str:
    """Return a file's SHA-256 digest, reading it in bounded chunks."""
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as error:
        raise OSError(f"Unable to read file for SHA-256: {file_path}: {error}") from error
    return digest.hexdigest()


def load_dataset_paths(config_path: Path, project_root: Path) -> tuple[Path, Path]:
    """Read the images and labels entries from the simple paths section in config.yaml."""
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OSError(f"Unable to read configuration file {config_path}: {error}") from error

    in_paths = False
    configured: dict[str, str] = {}
    for line in lines:
        content = line.split("#", maxsplit=1)[0].rstrip()
        if not content.strip():
            continue
        if not content[0].isspace():
            in_paths = content.strip() == "paths:"
            continue
        if in_paths:
            key, separator, value = content.strip().partition(":")
            if separator and key in {"images", "labels"}:
                configured[key] = value.strip().strip("'\"")

    if not configured.get("images") or not configured.get("labels"):
        raise ValueError(
            f"Configuration file {config_path} must define paths.images and paths.labels."
        )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    return resolve(configured["images"]), resolve(configured["labels"])


def _list_files(
    directory: Path,
    name: str,
    result: DataCheckResult,
) -> list[Path]:
    """Return files below a directory, recording inaccessible paths as operational errors."""
    if not directory.exists():
        result.warnings.append(f"{name} directory is absent: {directory}")
        return []
    if not directory.is_dir():
        message = f"{name} path is not a directory: {directory}"
        result.errors.append(message)
        result.operational_errors.append(message)
        return []

    files: list[Path] = []

    def record_walk_error(error: OSError) -> None:
        message = f"Unable to scan {name} directory: {error}"
        result.errors.append(message)
        result.operational_errors.append(message)

    for current_directory, _, filenames in os.walk(
        directory, onerror=record_walk_error
    ):
        files.extend(
            Path(current_directory) / filename
            for filename in filenames
            if filename != ".gitkeep"
        )
    return sorted(files)


def _validate_image(image_path: Path) -> str | None:
    """Decode a supported image and return a reason when it is invalid."""
    if image_path.stat().st_size == 0:
        return "file is empty"
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return f"unsupported image file extension: {image_path.suffix or '(none)'}"

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise ImageDependencyError(
            "Image validation requires Pillow. Install it with "
            "'py -3.12 -m pip install -r requirements.txt'."
        ) from error

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as error:
        return f"corrupt or unreadable image: {error}"
    except Image.DecompressionBombError as error:
        return f"image exceeds Pillow's safe decoding limits: {error}"
    return None


def _validate_label_line(line: str) -> str | None:
    """Return a validation message for one YOLO detection label, if invalid."""
    fields = line.split()
    if len(fields) != 5:
        return "expected exactly five fields: class_id x_center y_center width height"

    try:
        class_id = int(fields[0])
    except ValueError:
        return "class_id must be a non-negative integer"
    if class_id < 0:
        return "class_id must be a non-negative integer"

    try:
        x_center, y_center, width, height = (float(value) for value in fields[1:])
    except ValueError:
        return "coordinates and dimensions must be numeric"
    if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
        return "coordinates and dimensions must be finite numbers"
    if not 0 <= x_center <= 1:
        return "x_center must be within [0, 1]"
    if not 0 <= y_center <= 1:
        return "y_center must be within [0, 1]"
    if not 0 < width <= 1:
        return "width must be within (0, 1]"
    if not 0 < height <= 1:
        return "height must be within (0, 1]"
    return None


def _relative_key(file_path: Path, root: Path) -> str:
    """Create a case-insensitive relative key for pairing image and label files."""
    return file_path.relative_to(root).as_posix().casefold()


def scan_dataset(image_directory: Path, label_directory: Path) -> DataCheckResult:
    """Check image files, byte-identical duplicates, and YOLO label pairings."""
    result = DataCheckResult()
    image_files = _list_files(image_directory, "Images", result)
    label_files = _list_files(label_directory, "Labels", result)

    yolo_label_files = [
        path for path in label_files if path.suffix.lower() == ".txt"
    ]
    supported_images = [
        path
        for path in image_files
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    result.data_available = bool(image_files or yolo_label_files)
    if not result.data_available:
        result.warnings.append(
            "No data available: the image and label directories contain no files."
        )

    hashes: dict[str, list[str]] = {}
    for image_path in image_files:
        result.images_examined += 1
        try:
            reason = _validate_image(image_path)
        except ImageDependencyError:
            raise
        except OSError as error:
            reason = f"unable to inspect image: {error}"

        if reason is None:
            result.valid_images += 1
        else:
            message = f"{image_path}: {reason}"
            result.invalid_images.append(message)
            result.errors.append(message)

        if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            try:
                digest = sha256_file(image_path)
            except OSError as error:
                message = str(error)
                result.errors.append(message)
                result.operational_errors.append(message)
            else:
                hashes.setdefault(digest, []).append(str(image_path))

    result.duplicate_groups = [
        paths for paths in hashes.values() if len(paths) > 1
    ]
    for group in result.duplicate_groups:
        result.warnings.append(f"Byte-identical duplicate images: {', '.join(group)}")

    image_keys = {
        _relative_key(path.with_suffix(".txt"), image_directory)
        for path in supported_images
    }
    label_keys: dict[str, Path] = {
        _relative_key(path, label_directory): path for path in yolo_label_files
    }
    result.images_missing_labels = [
        str(path)
        for path in supported_images
        if _relative_key(path.with_suffix(".txt"), image_directory) not in label_keys
    ]
    result.labels_missing_images = [
        str(path)
        for path in yolo_label_files
        if _relative_key(path, label_directory) not in image_keys
    ]
    for path in result.images_missing_labels:
        result.warnings.append(f"Image has no corresponding label file: {path}")
    for path in result.labels_missing_images:
        result.warnings.append(f"Label has no corresponding image: {path}")

    for label_path in yolo_label_files:
        try:
            contents = label_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            message = f"{label_path}: unable to read label file: {error}"
            result.invalid_label_files.append(message)
            result.errors.append(message)
            continue

        line_errors: list[str] = []
        for line_number, line in enumerate(contents.splitlines(), start=1):
            if not line.strip():
                continue
            problem = _validate_label_line(line)
            if problem:
                line_errors.append(f"{label_path}:{line_number}: {problem}")

        if line_errors:
            result.invalid_label_files.append(str(label_path))
            result.errors.extend(line_errors)
        else:
            result.valid_label_files += 1
            if not contents.strip():
                result.empty_label_files.append(str(label_path))
                result.warnings.append(
                    f"Empty label file is valid and represents zero objects: {label_path}"
                )

    return result


def format_data_check(result: DataCheckResult) -> str:
    """Format dataset findings as a readable CLI summary."""
    lines = [
        "Data integrity check",
        f"Data availability: {'data found' if result.data_available else 'no data available'}",
        f"Images examined: {result.images_examined}",
        f"Valid images: {result.valid_images}",
        f"Invalid images: {len(result.invalid_images)}",
        f"Duplicate groups: {len(result.duplicate_groups)}",
        f"Valid label files: {result.valid_label_files}",
        f"Invalid label files: {len(result.invalid_label_files)}",
        f"Missing image-label pairs: {len(result.images_missing_labels) + len(result.labels_missing_images)}",
        f"Errors: {len(result.errors)}",
        f"Warnings: {len(result.warnings)}",
    ]
    for heading, messages in (
        ("Error details", result.errors),
        ("Warning details", result.warnings),
    ):
        if messages:
            lines.append(f"{heading}:")
            lines.extend(f"- {message}" for message in messages)
    return "\n".join(lines)
