"""Temporary-file tests for model baseline registration and verification."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.model.integrity import (
    ModelInferenceBlocked,
    ModelRegistryError,
    ensure_model_verified,
    register_model,
    verify_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "assurance.py"


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.model_file = self.root / "demo-model.bin"
        self.model_file.write_bytes(b"synthetic test bytes, not a pretrained model")
        self.registry_path = self.root / "registry" / "models.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_registers_readable_model_file(self) -> None:
        registration = register_model("demo-model", self.model_file, self.registry_path)

        self.assertEqual(
            registration["expected_hash"],
            hashlib.sha256(self.model_file.read_bytes()).hexdigest(),
        )
        self.assertEqual(registration["model_id"], "demo-model")
        self.assertTrue(self.registry_path.is_file())
        self.assertTrue(registration["registered_at"])

    def test_matching_registered_hash_is_verified(self) -> None:
        register_model("demo-model", self.model_file, self.registry_path)

        result = verify_model("demo-model", self.registry_path)

        self.assertEqual(result.status, "VERIFIED")
        self.assertEqual(result.expected_hash, result.actual_hash)

    def test_modified_model_file_is_suspicious(self) -> None:
        register_model("demo-model", self.model_file, self.registry_path)
        self.model_file.write_bytes(b"modified synthetic test bytes")

        result = verify_model("demo-model", self.registry_path)

        self.assertEqual(result.status, "SUSPICIOUS")
        self.assertNotEqual(result.expected_hash, result.actual_hash)
        self.assertIn("CRITICAL", result.reason)

    def test_missing_registered_model_requires_review(self) -> None:
        register_model("demo-model", self.model_file, self.registry_path)
        self.model_file.unlink()

        result = verify_model("demo-model", self.registry_path)

        self.assertEqual(result.status, "REVIEW REQUIRED")
        self.assertIsNotNone(result.expected_hash)
        self.assertIsNone(result.actual_hash)

    def test_unknown_id_and_missing_registry_require_review(self) -> None:
        unknown_result = verify_model("unknown", self.registry_path)
        missing_registry_result = verify_model(
            "demo-model", self.root / "absent" / "registry.json"
        )

        self.assertEqual(unknown_result.status, "REVIEW REQUIRED")
        self.assertIsNone(unknown_result.expected_hash)
        self.assertEqual(missing_registry_result.status, "REVIEW REQUIRED")

    def test_duplicate_identifier_requires_explicit_replacement(self) -> None:
        original = register_model("demo-model", self.model_file, self.registry_path)
        self.model_file.write_bytes(b"replacement synthetic bytes")

        with self.assertRaisesRegex(ModelRegistryError, "Use --replace"):
            register_model("demo-model", self.model_file, self.registry_path)

        replacement = register_model(
            "demo-model", self.model_file, self.registry_path, replace=True
        )
        self.assertNotEqual(original["expected_hash"], replacement["expected_hash"])

    def test_missing_model_file_cannot_be_registered(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            register_model(
                "demo-model", self.root / "missing.bin", self.registry_path
            )

    def test_malformed_registry_requires_review_and_blocks_registration(self) -> None:
        self.registry_path.parent.mkdir(parents=True)
        self.registry_path.write_text("{not valid json", encoding="utf-8")

        result = verify_model("demo-model", self.registry_path)

        self.assertEqual(result.status, "REVIEW REQUIRED")
        self.assertIn("Unable to read model registry", result.reason)
        with self.assertRaisesRegex(ModelRegistryError, "Unable to read model registry"):
            register_model("demo-model", self.model_file, self.registry_path)

    def test_duplicate_json_model_identifiers_are_rejected(self) -> None:
        self.registry_path.parent.mkdir(parents=True)
        self.registry_path.write_text(
            '{"schema_version":1,"models":{"demo-model":{}, "demo-model":{}}}',
            encoding="utf-8",
        )

        result = verify_model("demo-model", self.registry_path)

        self.assertEqual(result.status, "REVIEW REQUIRED")
        self.assertIn("Duplicate key", result.reason)

    def test_inference_guard_blocks_suspicious_model_before_operation(self) -> None:
        register_model("demo-model", self.model_file, self.registry_path)
        self.model_file.write_bytes(b"tampered")
        result = verify_model("demo-model", self.registry_path)
        operation_called = False

        def inference_operation() -> None:
            nonlocal operation_called
            operation_called = True

        with self.assertRaises(ModelInferenceBlocked):
            ensure_model_verified(result)
            inference_operation()

        self.assertFalse(operation_called)

    def test_cli_register_and_verify_commands(self) -> None:
        register_result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "register-model",
                "--id",
                "demo-model",
                "--path",
                str(self.model_file),
                "--registry",
                str(self.registry_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        verify_result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "verify-model",
                "--id",
                "demo-model",
                "--registry",
                str(self.registry_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(register_result.returncode, 0, register_result.stderr)
        self.assertIn("Expected SHA-256:", register_result.stdout)
        self.assertEqual(verify_result.returncode, 0, verify_result.stderr)
        self.assertIn("Status: VERIFIED", verify_result.stdout)
        self.assertIn("Actual SHA-256:", verify_result.stdout)


if __name__ == "__main__":
    unittest.main()
