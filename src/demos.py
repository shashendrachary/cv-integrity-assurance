"""Safe temporary-file demonstrations of project integrity checks."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from src.integrity.data_checks import format_data_check, scan_dataset
from src.integrity.data_checks import sha256_file
from src.model.integrity import register_model, verify_model
from src.vision.inference import run_inference


def run_model_tampering_demo(
    model_id: str,
    registry_path: Path,
    *,
    project_root: Path | None = None,
) -> tuple[str, str, str, bool]:
    """Tamper with a temporary copy only and demonstrate inference blocking."""
    original = verify_model(model_id, registry_path, project_root=project_root)
    if original.status != "VERIFIED" or original.file_path is None:
        raise ValueError(
            f"Demo requires an intact registered model; current status is {original.status}."
        )

    source_path = Path(original.file_path)
    source_digest = sha256_file(source_path)
    with tempfile.TemporaryDirectory(prefix="cv-integrity-demo-") as temporary_directory:
        demo_root = Path(temporary_directory)
        demo_model = demo_root / source_path.name
        shutil.copyfile(source_path, demo_model)
        demo_registry = demo_root / "registry.json"
        register_model("tamper-demo", demo_model, demo_registry)
        expected_hash = verify_model("tamper-demo", demo_registry).expected_hash

        with demo_model.open("ab") as model_file:
            model_file.write(b"\nDEMO-ONLY-TAMPER")
        result = verify_model("tamper-demo", demo_registry)

        demo_image = demo_root / "valid-input.png"
        Image.new("RGB", (32, 32), (90, 120, 150)).save(demo_image)
        loader = Mock(side_effect=AssertionError("tampered model must not load"))
        inference = run_inference(
            demo_image,
            "tamper-demo",
            demo_registry,
            model_loader=loader,
        )
        blocked_before_load = (
            inference.status == "SUSPICIOUS" and not loader.called
        )

        print(f"Demo model copy: {demo_model}")
        print(f"Expected SHA-256: {expected_hash}")
        print(f"Actual SHA-256:   {result.actual_hash}")
        print(f"Verification status: {result.status}")
        print(
            "Inference loading: "
            + ("BLOCKED before loader" if blocked_before_load else "NOT BLOCKED")
        )

        expected = expected_hash or ""
        actual = result.actual_hash or ""
        status = result.status
    finally_digest = sha256_file(source_path)
    if finally_digest != source_digest:
        raise RuntimeError("Original registered model changed during the demo.")
    if status != "SUSPICIOUS" or not blocked_before_load:
        raise RuntimeError("Model-tampering demo failed to block the test model.")
    return expected, actual, status, blocked_before_load


def run_data_integrity_demo() -> str:
    """Create isolated synthetic image/label examples and print their findings."""
    with tempfile.TemporaryDirectory(prefix="cv-integrity-data-demo-") as temporary_directory:
        root = Path(temporary_directory)
        images = root / "images"
        labels = root / "labels"
        images.mkdir()
        labels.mkdir()

        valid_path = images / "valid.png"
        Image.new("RGB", (32, 24), (80, 120, 160)).save(valid_path)
        (images / "duplicate.png").write_bytes(valid_path.read_bytes())
        (images / "corrupt.png").write_bytes(b"not a valid image")
        Image.new("RGB", (24, 24), (110, 110, 110)).save(images / "missing-label.png")
        (labels / "valid.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
        )
        (labels / "corrupt.txt").write_text(
            "0 not-a-number 0.5 0.4 0.4\n", encoding="utf-8"
        )
        result = scan_dataset(images, labels)
        report = format_data_check(result)
        print("Temporary data-integrity demonstration (all files removed afterward):")
        print(report)
        return report
