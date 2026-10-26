"""Temporary-file tests for model baseline registration and verification."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import assurance
from src.integrity.data_checks import DataCheckResult
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

    def test_two_model_ids_keep_independent_registrations_and_verify(self) -> None:
        first_model = self.root / "first-model.bin"
        second_model = self.root / "second-model.bin"
        first_model.write_bytes(b"first model bytes")
        second_model.write_bytes(b"second model bytes")

        first_registration = register_model(
            "first-model", first_model, self.registry_path
        )
        second_registration = register_model(
            "second-model", second_model, self.registry_path
        )
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(registry["models"]), {"first-model", "second-model"}
        )
        self.assertEqual(
            registry["models"]["first-model"], first_registration
        )
        self.assertEqual(
            registry["models"]["second-model"], second_registration
        )
        self.assertEqual(
            verify_model("first-model", self.registry_path).status, "VERIFIED"
        )
        self.assertEqual(
            verify_model("second-model", self.registry_path).status, "VERIFIED"
        )

    def test_verification_does_not_change_any_registration(self) -> None:
        first_model = self.root / "first-model.bin"
        second_model = self.root / "second-model.bin"
        first_model.write_bytes(b"first model bytes")
        second_model.write_bytes(b"second model bytes")
        register_model("first-model", first_model, self.registry_path)
        register_model("second-model", second_model, self.registry_path)
        original_registry = self.registry_path.read_bytes()

        self.assertEqual(
            verify_model("first-model", self.registry_path).status, "VERIFIED"
        )

        self.assertEqual(self.registry_path.read_bytes(), original_registry)

    def test_replacing_one_model_id_preserves_the_other_registration(self) -> None:
        first_model = self.root / "first-model.bin"
        first_replacement = self.root / "first-model-replacement.bin"
        second_model = self.root / "second-model.bin"
        first_model.write_bytes(b"first model bytes")
        first_replacement.write_bytes(b"replacement model bytes")
        second_model.write_bytes(b"second model bytes")
        original_first = register_model(
            "first-model", first_model, self.registry_path
        )
        original_second = register_model(
            "second-model", second_model, self.registry_path
        )

        with self.assertRaisesRegex(ModelRegistryError, "Use --replace"):
            register_model("first-model", first_replacement, self.registry_path)

        replacement = register_model(
            "first-model",
            first_replacement,
            self.registry_path,
            replace=True,
        )
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))

        self.assertNotEqual(
            replacement["expected_hash"], original_first["expected_hash"]
        )
        self.assertEqual(registry["models"]["first-model"], replacement)
        self.assertEqual(registry["models"]["second-model"], original_second)
        self.assertEqual(
            verify_model("first-model", self.registry_path).expected_hash,
            replacement["expected_hash"],
        )
        self.assertEqual(
            verify_model("second-model", self.registry_path).expected_hash,
            original_second["expected_hash"],
        )

    def test_legacy_yolo11n_model_id_remains_supported(self) -> None:
        registration = register_model(
            "yolo11n", self.model_file, self.registry_path
        )

        result = verify_model("yolo11n", self.registry_path)

        self.assertEqual(result.status, "VERIFIED")
        self.assertEqual(result.model_id, "yolo11n")
        self.assertEqual(result.expected_hash, registration["expected_hash"])

    def test_invalid_model_id_is_rejected_without_changing_registry(self) -> None:
        registration = register_model(
            "valid-model", self.model_file, self.registry_path
        )
        original_registry = self.registry_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "Model ID must start"):
            register_model("../invalid", self.model_file, self.registry_path)

        result = verify_model("../invalid", self.registry_path)

        self.assertEqual(result.status, "REVIEW REQUIRED")
        self.assertIn("Model ID must start", result.reason)
        self.assertEqual(self.registry_path.read_bytes(), original_registry)
        self.assertEqual(
            json.loads(original_registry)["models"]["valid-model"],
            registration,
        )

    def test_relative_model_path_resolves_from_project_root_outside_cwd(self) -> None:
        project_root = self.root / "project"
        model_path = project_root / "models" / "demo-model.bin"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"project-relative model bytes")
        registry_path = model_path.parent / "model_registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": {
                        "demo-model": {
                            "model_id": "demo-model",
                            "file_path": "models/demo-model.bin",
                            "expected_hash": hashlib.sha256(
                                model_path.read_bytes()
                            ).hexdigest(),
                            "registered_at": "2026-01-01T00:00:00+00:00",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        unrelated_directory = self.root / "unrelated"
        unrelated_directory.mkdir()
        original_directory = Path.cwd()

        try:
            os.chdir(unrelated_directory)
            result = verify_model(
                "demo-model", registry_path, project_root=project_root
            )
        finally:
            os.chdir(original_directory)

        self.assertEqual(result.status, "VERIFIED")
        self.assertEqual(Path(result.file_path or ""), model_path.resolve())

    def test_absolute_model_path_remains_supported(self) -> None:
        register_model("demo-model", self.model_file, self.registry_path)

        result = verify_model(
            "demo-model", self.registry_path, project_root=self.root / "other-root"
        )

        self.assertEqual(result.status, "VERIFIED")
        self.assertEqual(Path(result.file_path or ""), self.model_file.resolve())

    def test_missing_relative_model_path_requires_actionable_review(self) -> None:
        project_root = self.root / "project"
        registry_path = self.root / "registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": {
                        "demo-model": {
                            "model_id": "demo-model",
                            "file_path": "models/missing.bin",
                            "expected_hash": hashlib.sha256(b"expected").hexdigest(),
                            "registered_at": "2026-01-01T00:00:00+00:00",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        result = verify_model(
            "demo-model", registry_path, project_root=project_root
        )

        self.assertEqual(result.status, "REVIEW REQUIRED")
        self.assertIn(str(project_root / "models" / "missing.bin"), result.reason)
        self.assertIn("Check the registry file_path", result.reason)

    def test_cli_resolves_registry_and_model_paths_outside_cwd(self) -> None:
        project_root = self.root / "project"
        model_path = project_root / "models" / "demo-model.bin"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"CLI-relative model bytes")
        registry_path = model_path.parent / "model_registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": {
                        "demo-model": {
                            "model_id": "demo-model",
                            "file_path": "models/demo-model.bin",
                            "expected_hash": hashlib.sha256(
                                model_path.read_bytes()
                            ).hexdigest(),
                            "registered_at": "2026-01-01T00:00:00+00:00",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (project_root / "config.yaml").write_text(
            "paths:\n  model_registry: models/model_registry.json\n",
            encoding="utf-8",
        )
        unrelated_directory = self.root / "unrelated"
        unrelated_directory.mkdir()
        output = io.StringIO()
        original_directory = Path.cwd()

        try:
            os.chdir(unrelated_directory)
            with (
                patch("assurance.PROJECT_ROOT", project_root),
                patch("assurance.configure_logging"),
                redirect_stdout(output),
            ):
                result = assurance.main(["verify-model", "--id", "demo-model"])
        finally:
            os.chdir(original_directory)

        self.assertEqual(result, 0)
        self.assertIn("Status: VERIFIED", output.getvalue())
        self.assertIn(f"File: {model_path.resolve()}", output.getvalue())

    def test_cli_resolves_relative_dataset_overrides_from_project_root(self) -> None:
        project_root = self.root / "project"
        project_root.mkdir()
        (project_root / "config.yaml").write_text(
            "paths:\n  images: configured-images\n  labels: configured-labels\n",
            encoding="utf-8",
        )
        unrelated_directory = self.root / "unrelated"
        unrelated_directory.mkdir()
        scan = patch(
            "assurance.scan_dataset", return_value=DataCheckResult()
        )
        original_directory = Path.cwd()

        try:
            os.chdir(unrelated_directory)
            with (
                patch("assurance.PROJECT_ROOT", project_root),
                patch("assurance.configure_logging"),
                scan as scan_dataset,
            ):
                result = assurance.main(
                    [
                        "check-data",
                        "--images-dir",
                        "override-images",
                        "--labels-dir",
                        "override-labels",
                    ]
                )
        finally:
            os.chdir(original_directory)

        self.assertEqual(result, 0)
        scan_dataset.assert_called_once_with(
            project_root / "override-images",
            project_root / "override-labels",
        )

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
