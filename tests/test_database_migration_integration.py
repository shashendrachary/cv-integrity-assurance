"""Opt-in PostgreSQL migration tests for one explicitly named disposable database."""

import os
import re
import unittest
from datetime import datetime, timezone
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

from src.database.postgres import (
    DatabaseOperationError,
    PROJECT_SCOPE_SCHEMA_VERSION,
    initialize_provenance_table,
    migrate_project_scope_schema,
)
from src.provenance.records import ProvenanceRecord, insert_provenance_record


TEST_DATABASE_NAME = "cv_integrity_migration_test"
TEST_DATABASE_URL_ENV = "ASSURANCE_TEST_DATABASE_URL"


def _validated_test_database_url(value: str | None) -> str:
    if not value:
        raise ValueError(f"{TEST_DATABASE_URL_ENV} must be set explicitly.")
    try:
        parsed = urlsplit(value)
        database_name = unquote(parsed.path.lstrip("/"))
        query = parse_qs(parsed.query, keep_blank_values=True)
    except ValueError:
        raise ValueError(
            f"{TEST_DATABASE_URL_ENV} must be a PostgreSQL URL."
        ) from None
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError(f"{TEST_DATABASE_URL_ENV} must be a PostgreSQL URL.")
    query_database_names = query.get("dbname", []) + query.get("database", [])
    if (
        database_name != TEST_DATABASE_NAME
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.port != 5433
        or any(name != TEST_DATABASE_NAME for name in query_database_names)
        or "service" in query
        or "options" in query
        or "host" in query
        or "hostaddr" in query
        or "port" in query
    ):
        raise ValueError(
            f"{TEST_DATABASE_URL_ENV} must target exactly "
            f"{TEST_DATABASE_NAME}; refusing to connect."
        )
    return value


class TestDatabaseUrlSafetyTests(unittest.TestCase):
    def test_missing_url_is_rejected_without_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be set explicitly"):
            _validated_test_database_url(None)

    def test_non_test_database_url_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to connect"):
            _validated_test_database_url(
                "postgresql://tester@localhost:5433/cv_integrity"
            )

    def test_non_local_endpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to connect"):
            _validated_test_database_url(
                "postgresql://tester@db.example.invalid:5433/"
                "cv_integrity_migration_test"
            )

    def test_unexpected_port_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to connect"):
            _validated_test_database_url(
                "postgresql://tester@localhost:5432/"
                "cv_integrity_migration_test"
            )

    def test_database_override_in_url_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to connect"):
            _validated_test_database_url(
                "postgresql://tester@localhost:5433/"
                "cv_integrity_migration_test?dbname=cv_integrity"
            )

    def test_connection_overrides_in_url_query_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to connect"):
            _validated_test_database_url(
                "postgresql://tester@localhost:5433/"
                "cv_integrity_migration_test?options=-csearch_path=public"
            )

    def test_host_address_override_in_url_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to connect"):
            _validated_test_database_url(
                "postgresql://tester@localhost:5433/"
                "cv_integrity_migration_test?hostaddr=203.0.113.1"
            )

    def test_exact_disposable_database_name_is_accepted(self) -> None:
        value = (
            "postgresql://tester@localhost:5433/"
            "cv_integrity_migration_test"
        )
        self.assertEqual(_validated_test_database_url(value), value)


class PostgreSQLProjectScopeMigrationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        url = os.environ.get(TEST_DATABASE_URL_ENV)
        if not url:
            raise unittest.SkipTest(
                f"Set {TEST_DATABASE_URL_ENV} to run isolated PostgreSQL tests."
            )
        cls.database_url = _validated_test_database_url(url)
        try:
            import psycopg
        except ImportError:
            raise unittest.SkipTest(
                "Psycopg 3 is required for PostgreSQL integration tests."
            ) from None
        cls.psycopg = psycopg

    def setUp(self) -> None:
        self.schema_name = f"assurance_it_{uuid.uuid4().hex}"
        if not re.fullmatch(r"assurance_it_[0-9a-f]{32}", self.schema_name):
            self.fail("Generated an invalid isolated test schema name.")
        try:
            with self.connect_test_database() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f'CREATE SCHEMA "{self.schema_name}"')
        except Exception:
            raise RuntimeError(
                "Could not create an isolated schema in the explicitly configured "
                f"{TEST_DATABASE_NAME} database."
            ) from None
        self.settings = {
            "host": "localhost",
            "port": 5433,
            "dbname": TEST_DATABASE_NAME,
            "user": "integration-test",
        }
        self.connection_factory = self.make_connection_factory()

    def tearDown(self) -> None:
        try:
            with self.connect_test_database() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f'DROP SCHEMA "{self.schema_name}" CASCADE')
        except Exception:
            raise RuntimeError(
                f"Could not remove isolated schema from {TEST_DATABASE_NAME}."
            ) from None

    def connect_test_database(self, options: str | None = None):
        try:
            connection_options = {
                "host": "localhost",
                "hostaddr": "127.0.0.1",
                "port": 5433,
            }
            if options is not None:
                connection_options["options"] = options
            connection = (
                self.psycopg.connect(self.database_url, **connection_options)
            )
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database()")
                if cursor.fetchone()[0] != TEST_DATABASE_NAME:
                    connection.close()
                    raise RuntimeError("Connected to an unexpected database.")
            connection.commit()
            return connection
        except Exception:
            raise RuntimeError(
                f"Could not connect to the explicitly configured "
                f"{TEST_DATABASE_NAME} database."
            ) from None

    def make_connection_factory(self):
        options = f"-c search_path={self.schema_name}"

        def connect(**_settings):
            return self.connect_test_database(options)

        return connect

    def test_empty_schema_migrates_and_has_expected_constraints_and_index(self) -> None:
        version = migrate_project_scope_schema(
            self.settings,
            connection_factory=self.connection_factory,
        )
        self.assertEqual(version, PROJECT_SCOPE_SCHEMA_VERSION)

        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT singleton, version
                    FROM assurance_schema_version
                    """
                )
                self.assertEqual(cursor.fetchone(), (True, PROJECT_SCOPE_SCHEMA_VERSION))

                cursor.execute(
                    """
                    SELECT organization_id, project_id
                    FROM projects
                    """
                )
                self.assertEqual(cursor.fetchall(), [("local", "default")])

                cursor.execute(
                    """
                    SELECT column_name, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'inference_provenance'
                      AND column_name IN ('organization_id', 'project_id')
                    ORDER BY column_name
                    """
                )
                self.assertEqual(
                    cursor.fetchall(),
                    [("organization_id", "NO"), ("project_id", "NO")],
                )

                cursor.execute(
                    """
                    SELECT conname, contype, pg_get_constraintdef(oid, TRUE)
                    FROM pg_constraint
                    WHERE connamespace = current_schema()::regnamespace
                    """
                )
                definitions = {
                    name: (kind, definition)
                    for name, kind, definition in cursor.fetchall()
                }
                self.assertEqual(
                    definitions["assurance_schema_version_singleton_check"],
                    ("c", "CHECK (singleton)"),
                )
                self.assertEqual(
                    definitions["assurance_schema_version_pkey"],
                    ("p", "PRIMARY KEY (singleton)"),
                )
                self.assertEqual(
                    definitions["organizations_pkey"],
                    ("p", "PRIMARY KEY (organization_id)"),
                )
                self.assertIn(
                    "PRIMARY KEY (organization_id, project_id)",
                    definitions["projects_pkey"][1],
                )
                self.assertIn(
                    "FOREIGN KEY (organization_id) REFERENCES organizations",
                    definitions["projects_organization_id_fkey"][1],
                )
                self.assertEqual(
                    definitions["inference_provenance_pkey"],
                    ("p", "PRIMARY KEY (run_id)"),
                )
                self.assertIn(
                    "FOREIGN KEY (organization_id, project_id)",
                    definitions["inference_provenance_project_scope_fkey"][1],
                )

                cursor.execute(
                    """
                    SELECT index_row.indisvalid, index_row.indisready,
                           index_row.indisunique, index_row.indpred IS NULL,
                           access_method.amname, index_row.indnkeyatts,
                           index_row.indnatts,
                           array_agg(attribute.attname ORDER BY key.position)
                    FROM pg_index AS index_row
                    JOIN pg_class AS table_relation
                      ON table_relation.oid = index_row.indrelid
                    JOIN pg_namespace AS table_schema
                      ON table_schema.oid = table_relation.relnamespace
                    JOIN pg_class AS index_relation
                      ON index_relation.oid = index_row.indexrelid
                    JOIN pg_am AS access_method
                      ON access_method.oid = index_relation.relam
                    JOIN LATERAL unnest(index_row.indkey)
                      WITH ORDINALITY AS key(attnum, position)
                      ON key.position <= index_row.indnkeyatts
                    JOIN pg_attribute AS attribute
                      ON attribute.attrelid = table_relation.oid
                     AND attribute.attnum = key.attnum
                    WHERE table_schema.nspname = current_schema()
                      AND index_relation.relname =
                          'inference_provenance_project_recorded_at_idx'
                    GROUP BY index_row.indisvalid, index_row.indisready,
                             index_row.indisunique,
                             (index_row.indpred IS NULL),
                             access_method.amname, index_row.indnkeyatts,
                             index_row.indnatts
                    """
                )
                index_row = cursor.fetchone()
                self.assertIsNotNone(index_row)
                self.assertEqual(
                    index_row,
                    (
                        True,
                        True,
                        False,
                        True,
                        "btree",
                        3,
                        3,
                        ["organization_id", "project_id", "recorded_at"],
                    ),
                )

        record = ProvenanceRecord(
            run_id="82e5641a-a37e-4a8b-a282-a031eb12cccb",
            recorded_at=datetime.now(timezone.utc),
            image_path="sample.png",
            image_sha256="a" * 64,
            model_id="integration-model",
            model_path="model.bin",
            model_expected_sha256="b" * 64,
            model_actual_sha256="b" * 64,
            model_verification_status="VERIFIED",
            confidence=0.5,
            detections=[],
            image_metrics={},
            configuration={},
            run_status="VERIFIED",
            inference_succeeded=True,
            reasons=[],
            organization_id="org-integration",
            project_id="project-integration",
        )
        insert_provenance_record(
            record,
            self.settings,
            connection_factory=self.connection_factory,
        )
        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT organization_id, project_id
                    FROM inference_provenance
                    WHERE run_id = %s
                    """,
                    (record.run_id,),
                )
                self.assertEqual(
                    cursor.fetchone(),
                    ("org-integration", "project-integration"),
                )
                cursor.execute(
                    """
                    SELECT organization_id, project_id
                    FROM projects
                    WHERE organization_id = %s AND project_id = %s
                    """,
                    ("org-integration", "project-integration"),
                )
                self.assertEqual(
                    cursor.fetchone(),
                    ("org-integration", "project-integration"),
                )

        self.assertEqual(
            migrate_project_scope_schema(
                self.settings,
                connection_factory=self.connection_factory,
            ),
            PROJECT_SCOPE_SCHEMA_VERSION,
        )

    def test_legacy_rows_are_backfilled_without_changing_evidence(self) -> None:
        initialize_provenance_table(
            self.settings,
            connection_factory=self.connection_factory,
        )
        original_run_id = "3c77794f-0b7c-47f7-b7f6-47d420b418e4"
        original_recorded_at = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        original_image_hash = "a" * 64
        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO inference_provenance (
                        run_id, recorded_at, image_path, image_sha256, model_id,
                        model_path, model_expected_sha256, model_actual_sha256,
                        model_verification_status, confidence, detections,
                        image_metrics, configuration, run_status,
                        inference_succeeded, reasons
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb
                    )
                    """,
                    (
                        original_run_id,
                        original_recorded_at,
                        "sample.png",
                        original_image_hash,
                        "model-a",
                        "model.bin",
                        "b" * 64,
                        "b" * 64,
                        "VERIFIED",
                        0.75,
                        '[{"class_id":1}]',
                        '{"width":32}',
                        '{"confidence":0.25}',
                        "VERIFIED",
                        True,
                        "[]",
                    ),
                )

        migrate_project_scope_schema(
            self.settings,
            connection_factory=self.connection_factory,
        )

        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT run_id::text, recorded_at, image_path, image_sha256,
                           model_id, model_path, model_expected_sha256,
                           model_actual_sha256, model_verification_status,
                           confidence, detections, image_metrics, configuration,
                           run_status, inference_succeeded, reasons,
                           organization_id, project_id
                    FROM inference_provenance
                    """
                )
                self.assertEqual(
                    cursor.fetchone(),
                    (
                        original_run_id,
                        original_recorded_at,
                        "sample.png",
                        original_image_hash,
                        "model-a",
                        "model.bin",
                        "b" * 64,
                        "b" * 64,
                        "VERIFIED",
                        0.75,
                        [{"class_id": 1}],
                        {"width": 32},
                        {"confidence": 0.25},
                        "VERIFIED",
                        True,
                        [],
                        "local",
                        "default",
                    ),
                )

    def test_conflicting_legacy_index_rolls_back_all_migration_ddl(self) -> None:
        initialize_provenance_table(
            self.settings,
            connection_factory=self.connection_factory,
        )
        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO inference_provenance (
                        run_id, recorded_at, image_path, model_id,
                        model_verification_status, detections, image_metrics,
                        configuration, run_status, inference_succeeded, reasons
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                        %s::jsonb, %s, %s, %s::jsonb
                    )
                    """,
                    (
                        "909b341b-d32e-4667-b508-a55bac3ab690",
                        datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
                        "rollback.png",
                        "model-a",
                        "VERIFIED",
                        "[]",
                        "{}",
                        "{}",
                        "VERIFIED",
                        True,
                        "[]",
                    ),
                )
                cursor.execute(
                    """
                    CREATE INDEX inference_provenance_project_recorded_at_idx
                    ON inference_provenance (image_path)
                    """
                )

        with self.assertRaisesRegex(DatabaseOperationError, "migration failed"):
            migrate_project_scope_schema(
                self.settings,
                connection_factory=self.connection_factory,
            )

        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                    ORDER BY table_name
                    """
                )
                self.assertEqual(cursor.fetchall(), [("inference_provenance",)])
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'inference_provenance'
                    """
                )
                self.assertNotIn(
                    "organization_id",
                    {row[0] for row in cursor.fetchall()},
                )
                cursor.execute(
                    """
                    SELECT indexdef
                    FROM pg_indexes
                    WHERE schemaname = current_schema()
                      AND indexname =
                          'inference_provenance_project_recorded_at_idx'
                    """
                )
                self.assertIn("(image_path)", cursor.fetchone()[0])
                cursor.execute("SELECT image_path FROM inference_provenance")
                self.assertEqual(cursor.fetchone(), ("rollback.png",))


if __name__ == "__main__":
    unittest.main()
