"""Offline tests for inference evidence records and PostgreSQL operations."""

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from src.database.postgres import (
    DatabaseNotConfigured,
    DatabaseOperationError,
    initialize_provenance_table,
    load_database_settings,
)
from src.model.integrity import register_model
from src.provenance.records import (
    create_provenance_record,
    insert_provenance_record,
)
from src.project_context import load_project_context
from src.vision.inference import run_inference


class ProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_path = self.root / "sample.png"
        Image.new("RGB", (32, 32), (60, 80, 100)).save(self.image_path)
        self.model_path = self.root / "synthetic.bin"
        self.model_path.write_bytes(b"synthetic unit-test model bytes")
        self.registry_path = self.root / "registry.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_result(self):
        register_model("test-model", self.model_path, self.registry_path)
        model = Mock()
        box = SimpleNamespace(
            cls=[SimpleNamespace(item=lambda: 1)],
            conf=[SimpleNamespace(item=lambda: 0.875)],
            xyxy=[SimpleNamespace(tolist=lambda: [1, 2, 20, 21])],
        )

        class OneBox:
            cls = box.cls
            conf = box.conf
            xyxy = box.xyxy

            def __len__(self) -> int:
                return 1

        model.predict.return_value = [
            SimpleNamespace(boxes=OneBox(), names={1: "bicycle"})
        ]
        result = run_inference(
            self.image_path,
            "test-model",
            self.registry_path,
            model_loader=Mock(return_value=model),
        )
        return result

    def test_record_captures_inference_results_and_json_values(self) -> None:
        result = self.make_result()
        configuration = {"confidence": 0.25, "device": "cpu"}

        record = create_provenance_record(self.image_path, result, configuration)

        self.assertTrue(record.run_id)
        self.assertEqual(record.recorded_at.tzinfo, timezone.utc)
        self.assertEqual(len(record.image_sha256 or ""), 64)
        self.assertEqual(record.model_verification_status, "VERIFIED")
        self.assertEqual(record.model_actual_sha256, record.model_expected_sha256)
        self.assertTrue(record.inference_succeeded)
        self.assertEqual(record.detections[0]["class_name"], "bicycle")
        self.assertEqual(record.image_metrics["width"], 32)
        self.assertEqual(
            json.loads(json.dumps(record.detections))[0]["box_xyxy"],
            [1.0, 2.0, 20.0, 21.0],
        )
        self.assertEqual(json.loads(json.dumps(record.configuration)), configuration)
        self.assertEqual(record.organization_id, "local")
        self.assertEqual(record.project_id, "default")

    def test_record_and_insert_use_selected_project_context(self) -> None:
        configuration_path = self.root / "project.yaml"
        configuration_path.write_text(
            "project:\n"
            "  organization_id: org-qa\n"
            "  project_id: vision-test\n"
            "paths:\n"
            "  images: images\n"
            "  labels: labels\n"
            "  model_registry: registry.json\n"
            "  reports: reports\n",
            encoding="utf-8",
        )
        context = load_project_context(configuration_path)
        record = create_provenance_record(
            self.image_path,
            self.make_result(),
            {"confidence": 0.3},
            context,
        )
        cursor = Mock()
        connection = Mock()
        connection.cursor.return_value = cursor

        insert_provenance_record(
            record,
            {"host": "localhost", "port": 5433, "dbname": "db", "user": "user"},
            connection_factory=Mock(return_value=connection),
        )

        statement, parameters = cursor.execute.call_args.args
        self.assertIn("organization_id, project_id", statement)
        self.assertIn("%s, %s", statement)
        self.assertEqual(parameters[-2:], ("org-qa", "vision-test"))

    def test_missing_evidence_is_left_null(self) -> None:
        result = run_inference(
            self.image_path,
            "unknown-model",
            self.registry_path,
        )
        missing_image = self.root / "not-present.png"

        record = create_provenance_record(
            missing_image, result, {"confidence": 0.25}
        )

        self.assertIsNone(record.image_sha256)
        self.assertIsNone(record.model_path)
        self.assertIsNone(record.model_expected_sha256)
        self.assertIsNone(record.model_actual_sha256)
        self.assertFalse(record.inference_succeeded)

    def test_insert_uses_parameters_and_json_serialization(self) -> None:
        record = create_provenance_record(
            self.image_path, self.make_result(), {"confidence": 0.3}
        )
        record = replace(record, image_path="x'); DROP TABLE inference_provenance; --")
        cursor = Mock()
        connection = Mock()
        connection.cursor.return_value = cursor
        factory = Mock(return_value=connection)

        insert_provenance_record(
            record,
            {"host": "localhost", "port": 5433, "dbname": "db", "user": "user"},
            connection_factory=factory,
        )

        statement, parameters = cursor.execute.call_args.args
        self.assertIn("%s::jsonb", statement)
        self.assertNotIn(record.image_path, statement)
        self.assertIn(record.image_path, parameters)
        statements = [
            call.args[0] for call in cursor.execute.call_args_list
        ]
        self.assertIn("INSERT INTO organizations", statements[0])
        self.assertIn("ON CONFLICT (organization_id) DO NOTHING", statements[0])
        self.assertIn("INSERT INTO projects", statements[1])
        self.assertIn(
            "ON CONFLICT (organization_id, project_id) DO NOTHING",
            statements[1],
        )
        self.assertEqual(cursor.execute.call_args_list[0].args[1], ("local",))
        self.assertEqual(
            cursor.execute.call_args_list[1].args[1],
            ("local", "default"),
        )
        self.assertIsInstance(json.loads(parameters[10]), list)
        self.assertIsInstance(json.loads(parameters[11]), dict)
        self.assertEqual(parameters[-2:], ("local", "default"))
        connection.commit.assert_called_once()
        connection.close.assert_called_once()

    def test_project_registration_rolls_back_when_provenance_insert_fails(self) -> None:
        record = create_provenance_record(
            self.image_path,
            self.make_result(),
            {"confidence": 0.3},
        )
        cursor = Mock()
        cursor.execute.side_effect = [None, None, RuntimeError("insert failed")]
        connection = Mock()
        connection.cursor.return_value = cursor

        with self.assertRaisesRegex(DatabaseOperationError, "transaction was rolled back"):
            insert_provenance_record(
                record,
                {"host": "localhost", "port": 5433, "dbname": "db", "user": "user"},
                connection_factory=Mock(return_value=connection),
            )

        self.assertEqual(cursor.execute.call_count, 3)
        connection.rollback.assert_called_once()
        connection.commit.assert_not_called()
        connection.close.assert_called_once()

    def test_connection_failure_is_actionable_without_echoing_secrets(self) -> None:
        def unavailable(**_settings: object) -> object:
            raise OSError("driver detail should not include credential values")

        record = create_provenance_record(
            self.image_path, self.make_result(), {"confidence": 0.25}
        )
        with self.assertRaisesRegex(DatabaseOperationError, "Could not connect"):
            insert_provenance_record(
                record,
                {
                    "host": "localhost",
                    "port": 5433,
                    "dbname": "db",
                    "user": "user",
                    "password": "never-display-this",
                },
                connection_factory=unavailable,
            )

    def test_schema_initialization_is_create_if_missing(self) -> None:
        cursor = Mock()
        connection = Mock()
        connection.cursor.return_value = cursor

        initialize_provenance_table(
            {"host": "localhost", "port": 5433, "dbname": "db", "user": "user"},
            connection_factory=Mock(return_value=connection),
        )

        statement = cursor.execute.call_args.args[0]
        self.assertIn("CREATE TABLE IF NOT EXISTS inference_provenance", statement)
        self.assertNotIn("organization_id", statement)
        self.assertNotIn("project_id", statement)
        self.assertNotIn("DROP TABLE", statement)
        connection.commit.assert_called_once()

    def test_database_settings_defaults_port_and_uses_environment(self) -> None:
        settings = load_database_settings(
            self.root / "absent.env",
            {"POSTGRES_DB": "assurance", "POSTGRES_USER": "tester"},
        )

        self.assertIsNotNone(settings)
        self.assertEqual(settings["host"], "localhost")
        self.assertEqual(settings["port"], 5433)
        self.assertNotIn("password", settings)

    def test_database_settings_report_unconfigured(self) -> None:
        self.assertIsNone(load_database_settings(self.root / "absent.env", {}))
        with self.assertRaises(DatabaseNotConfigured):
            load_database_settings(
                self.root / "absent.env", {"POSTGRES_USER": "tester"}
            )


if __name__ == "__main__":
    unittest.main()
