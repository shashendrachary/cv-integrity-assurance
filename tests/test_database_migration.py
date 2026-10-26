"""Offline tests for project-scope schema migration decisions and transactions."""

from dataclasses import replace
import unittest
from unittest.mock import Mock

from src.database.postgres import (
    LEGACY_PROVENANCE_COLUMNS,
    MigrationStateError,
    ORGANIZATIONS_TABLE,
    PROJECTS_TABLE,
    PROJECT_SCOPE_FOREIGN_KEY,
    PROJECT_SCOPE_INDEX,
    PROJECT_SCOPE_SCHEMA_VERSION,
    PROVENANCE_TABLE,
    SchemaIndex,
    SchemaConstraint,
    ProjectScopeSchema,
    VERSION_TABLE,
    _inspect_project_scope_schema,
    migrate_project_scope_schema,
    _plan_project_scope_migration,
    _validate_current_project_schema,
)


def legacy_schema() -> ProjectScopeSchema:
    return ProjectScopeSchema(
        tables=frozenset({PROVENANCE_TABLE}),
        columns={
            PROVENANCE_TABLE: dict(LEGACY_PROVENANCE_COLUMNS),
        },
        constraints=(
            SchemaConstraint(
                PROVENANCE_TABLE,
                "inference_provenance_pkey",
                "PRIMARY KEY",
                ("run_id",),
                definition="PRIMARY KEY (run_id)",
            ),
        ),
        indexes={},
        version=None,
        version_table_exists=False,
    )


def current_schema(version: int = PROJECT_SCOPE_SCHEMA_VERSION) -> ProjectScopeSchema:
    constraints = (
        SchemaConstraint(
            VERSION_TABLE,
            "assurance_schema_version_pkey",
            "PRIMARY KEY",
            ("singleton",),
            definition="PRIMARY KEY (singleton)",
        ),
        SchemaConstraint(
            VERSION_TABLE,
            "assurance_schema_version_singleton_check",
            "CHECK",
            (),
            definition="CHECK (singleton)",
        ),
        SchemaConstraint(
            ORGANIZATIONS_TABLE,
            "organizations_pkey",
            "PRIMARY KEY",
            ("organization_id",),
            definition="PRIMARY KEY (organization_id)",
        ),
        SchemaConstraint(
            PROJECTS_TABLE,
            "projects_pkey",
            "PRIMARY KEY",
            ("organization_id", "project_id"),
            definition="PRIMARY KEY (organization_id, project_id)",
        ),
        SchemaConstraint(
            PROJECTS_TABLE,
            "projects_organization_id_fkey",
            "FOREIGN KEY",
            ("organization_id",),
            ORGANIZATIONS_TABLE,
            ("organization_id",),
            "FOREIGN KEY (organization_id) REFERENCES organizations(organization_id)",
        ),
        SchemaConstraint(
            PROVENANCE_TABLE,
            "inference_provenance_pkey",
            "PRIMARY KEY",
            ("run_id",),
            definition="PRIMARY KEY (run_id)",
        ),
        SchemaConstraint(
            PROVENANCE_TABLE,
            PROJECT_SCOPE_FOREIGN_KEY,
            "FOREIGN KEY",
            ("organization_id", "project_id"),
            PROJECTS_TABLE,
            ("organization_id", "project_id"),
            "FOREIGN KEY (organization_id, project_id) "
            "REFERENCES projects(organization_id, project_id)",
        ),
    )
    return ProjectScopeSchema(
        tables=frozenset(
            {VERSION_TABLE, ORGANIZATIONS_TABLE, PROJECTS_TABLE, PROVENANCE_TABLE}
        ),
        columns={
            VERSION_TABLE: {"singleton": ("bool", True), "version": ("int4", True)},
            ORGANIZATIONS_TABLE: {"organization_id": ("text", True)},
            PROJECTS_TABLE: {
                "organization_id": ("text", True),
                "project_id": ("text", True),
            },
            PROVENANCE_TABLE: {
                **LEGACY_PROVENANCE_COLUMNS,
                "organization_id": ("text", True),
                "project_id": ("text", True),
            },
        },
        constraints=constraints,
        indexes={
            PROJECT_SCOPE_INDEX: SchemaIndex(
                access_method="btree",
                valid=True,
                ready=True,
                unique=False,
                predicate_free=True,
                key_columns=("organization_id", "project_id", "recorded_at"),
                total_columns=3,
                plain_ascending=True,
                default_operator_classes=True,
            )
        },
        version=version,
        version_table_exists=True,
        default_project_exists=True,
    )


class MigrationCursor:
    def __init__(
        self,
        *,
        tables: list[tuple[str, str]],
        columns: dict[str, list[tuple[str, str, str]]],
        constraints: list[tuple[object, ...]],
        indexes: list[tuple[object, ...]],
        version_rows: list[tuple[int]] | None = None,
        default_project_count: int = 1,
        unassigned_count: int = 0,
        partial_scope_count: int = 0,
        index_metadata: list[tuple[object, ...]] | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.tables = tables
        self.columns = columns
        self.constraints = constraints
        self.indexes = indexes
        self.version_rows = version_rows or []
        self.default_project_count = default_project_count
        self.unassigned_count = unassigned_count
        self.partial_scope_count = partial_scope_count
        self.index_metadata = index_metadata if index_metadata is not None else indexes
        self.fail_on = fail_on
        self.statements: list[tuple[str, tuple[object, ...] | None]] = []
        self.rows: list[tuple[object, ...]] = []
        self.closed = False

    def execute(
        self,
        statement: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        self.statements.append((statement, parameters))
        if self.fail_on and self.fail_on in statement:
            raise RuntimeError("simulated migration failure")
        normalized_statement = " ".join(statement.split())
        if "FROM information_schema.tables" in statement:
            self.rows = self.tables
        elif "FROM information_schema.columns" in statement:
            self.rows = self.columns.get(parameters[0], []) if parameters else []
        elif "FROM pg_constraint" in statement:
            self.rows = self.constraints
        elif "FROM pg_index" in statement:
            self.rows = self.index_metadata
        elif statement.strip().startswith(f"SELECT version FROM {VERSION_TABLE}"):
            self.rows = self.version_rows
        elif normalized_statement.startswith(f"SELECT COUNT(*) FROM {PROJECTS_TABLE}"):
            self.rows = [(self.default_project_count,)]
        elif normalized_statement.startswith(f"SELECT COUNT(*) FROM {PROVENANCE_TABLE}"):
            self.rows = [
                (
                    self.partial_scope_count or self.unassigned_count
                    if "OR project_id IS NULL" in normalized_statement
                    else self.unassigned_count,
                )
            ]
        else:
            self.rows = []

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows

    def fetchone(self) -> tuple[object, ...]:
        return self.rows[0]

    def close(self) -> None:
        self.closed = True


def legacy_metadata() -> tuple[
    list[tuple[str, str]],
    dict[str, list[tuple[str, str, str]]],
    list[tuple[object, ...]],
]:
    return (
        [(PROVENANCE_TABLE, "BASE TABLE")],
        {
            PROVENANCE_TABLE: [
                (name, data_type, "NO" if required else "YES")
                for name, (data_type, required) in LEGACY_PROVENANCE_COLUMNS.items()
            ]
        },
        [
            (
                PROVENANCE_TABLE,
                "inference_provenance_pkey",
                "p",
                "run_id",
                None,
                None,
                "PRIMARY KEY (run_id)",
                True,
            )
        ],
    )


def current_metadata() -> tuple[
    list[tuple[str, str]],
    dict[str, list[tuple[str, str, str]]],
    list[tuple[object, ...]],
    list[tuple[object, ...]],
]:
    schema = current_schema()
    table_rows = [(name, "BASE TABLE") for name in sorted(schema.tables)]
    column_rows = {
        table: [
            (name, data_type, "NO" if required else "YES")
            for name, (data_type, required) in columns.items()
        ]
        for table, columns in schema.columns.items()
    }
    constraint_rows: list[tuple[object, ...]] = []
    constraint_types = {
        "PRIMARY KEY": "p",
        "FOREIGN KEY": "f",
        "CHECK": "c",
    }
    for constraint in schema.constraints:
        if constraint.columns:
            constraint_rows.extend(
                (
                    constraint.table,
                    constraint.name,
                    constraint_types[constraint.kind],
                    column,
                    constraint.referenced_table,
                    ref_column,
                    constraint.definition,
                    constraint.validated,
                )
                for column, ref_column in zip(
                    constraint.columns,
                    constraint.referenced_columns or (None,) * len(constraint.columns),
                )
            )
        else:
            constraint_rows.append(
                (
                    constraint.table,
                    constraint.name,
                    constraint_types[constraint.kind],
                    None,
                    None,
                    None,
                    constraint.definition,
                    constraint.validated,
                )
            )
    index_metadata = [
        (
            name,
            index.access_method,
            index.valid,
            index.ready,
            index.unique,
            index.predicate_free,
            len(index.key_columns),
            index.total_columns,
            list(index.key_columns),
            index.plain_ascending,
            index.default_operator_classes,
        )
        for name, index in schema.indexes.items()
    ]
    return (
        table_rows,
        column_rows,
        constraint_rows,
        index_metadata,
    )


class ProjectScopeMigrationTests(unittest.TestCase):
    def test_plans_empty_legacy_and_current_schemas(self) -> None:
        empty = ProjectScopeSchema(
            tables=frozenset(),
            columns={},
            constraints=(),
            indexes={},
            version=None,
            version_table_exists=False,
        )

        self.assertEqual(_plan_project_scope_migration(empty), "new")
        self.assertEqual(_plan_project_scope_migration(legacy_schema()), "legacy")
        self.assertEqual(_plan_project_scope_migration(current_schema()), "current")

    def test_unknown_version_and_partial_schema_fail_safely(self) -> None:
        with self.assertRaisesRegex(MigrationStateError, "newer than supported"):
            _plan_project_scope_migration(current_schema(version=99))

        partial = ProjectScopeSchema(
            tables=frozenset({PROVENANCE_TABLE}),
            columns={
                PROVENANCE_TABLE: {
                    **LEGACY_PROVENANCE_COLUMNS,
                    "organization_id": ("text", False),
                }
            },
            constraints=legacy_schema().constraints,
            indexes={},
            version=None,
            version_table_exists=False,
        )
        with self.assertRaisesRegex(MigrationStateError, "manual review"):
            _plan_project_scope_migration(partial)

        orphan_projects = ProjectScopeSchema(
            tables=frozenset({PROJECTS_TABLE}),
            columns={},
            constraints=(),
            indexes={},
            version=None,
            version_table_exists=False,
        )
        with self.assertRaisesRegex(MigrationStateError, "manual review"):
            _plan_project_scope_migration(orphan_projects)

    def test_current_schema_validation_checks_constraint_and_index_definitions(self) -> None:
        schema = current_schema()
        invalid_check = tuple(
            replace(constraint, definition="CHECK (singleton IS TRUE)")
            if constraint.name == "assurance_schema_version_singleton_check"
            else constraint
            for constraint in schema.constraints
        )
        invalid_constraint_schema = replace(schema, constraints=invalid_check)
        with self.assertRaisesRegex(MigrationStateError, "required constraints"):
            _validate_current_project_schema(invalid_constraint_schema)

        invalid_foreign_key = tuple(
            replace(constraint, validated=False)
            if constraint.name == PROJECT_SCOPE_FOREIGN_KEY
            else constraint
            for constraint in schema.constraints
        )
        with self.assertRaisesRegex(MigrationStateError, "required constraints"):
            _validate_current_project_schema(
                replace(schema, constraints=invalid_foreign_key)
            )

        incompatible_indexes = (
            replace(schema.indexes[PROJECT_SCOPE_INDEX], predicate_free=False),
            replace(schema.indexes[PROJECT_SCOPE_INDEX], valid=False),
            replace(schema.indexes[PROJECT_SCOPE_INDEX], access_method="hash"),
            replace(schema.indexes[PROJECT_SCOPE_INDEX], total_columns=4),
            replace(schema.indexes[PROJECT_SCOPE_INDEX], plain_ascending=False),
            replace(
                schema.indexes[PROJECT_SCOPE_INDEX],
                default_operator_classes=False,
            ),
            replace(
                schema.indexes[PROJECT_SCOPE_INDEX],
                key_columns=("project_id", "organization_id", "recorded_at"),
            ),
        )
        for index in incompatible_indexes:
            with self.subTest(index=index):
                with self.assertRaisesRegex(MigrationStateError, "scoped provenance index"):
                    _validate_current_project_schema(
                        replace(
                            schema,
                            indexes={PROJECT_SCOPE_INDEX: index},
                        )
                    )

    def test_catalog_inspection_separates_same_named_constraints_by_table(self) -> None:
        tables, columns, _, indexes = current_metadata()
        constraints = [
            (
                ORGANIZATIONS_TABLE,
                "shared_fkey",
                "f",
                "organization_id",
                PROJECTS_TABLE,
                "organization_id",
                "FOREIGN KEY (organization_id) REFERENCES projects(organization_id)",
                True,
            ),
            (
                PROJECTS_TABLE,
                "shared_fkey",
                "f",
                "organization_id",
                ORGANIZATIONS_TABLE,
                "organization_id",
                "FOREIGN KEY (organization_id) REFERENCES organizations(organization_id)",
                True,
            ),
        ]
        cursor = MigrationCursor(
            tables=tables,
            columns=columns,
            constraints=constraints,
            indexes=indexes,
            version_rows=[(PROJECT_SCOPE_SCHEMA_VERSION,)],
        )

        schema = _inspect_project_scope_schema(cursor)
        inspected = {
            constraint.table: constraint
            for constraint in schema.constraints
        }
        self.assertEqual(
            inspected[ORGANIZATIONS_TABLE].referenced_table,
            PROJECTS_TABLE,
        )
        self.assertEqual(
            inspected[PROJECTS_TABLE].referenced_table,
            ORGANIZATIONS_TABLE,
        )
        constraint_query = next(
            statement
            for statement, _ in cursor.statements
            if "FROM pg_constraint" in statement
        )
        self.assertIn("source.oid = constraint_row.conrelid", constraint_query)
        self.assertIn("source_schema.nspname = current_schema()", constraint_query)

    def make_connection(self, cursor: MigrationCursor) -> Mock:
        connection = Mock()
        connection.cursor.return_value = cursor
        return connection

    def test_new_schema_migration_commits_with_parameterized_version(self) -> None:
        cursor = MigrationCursor(
            tables=[],
            columns={},
            constraints=[],
            indexes=[],
        )
        connection = self.make_connection(cursor)

        version = migrate_project_scope_schema(
            {"host": "localhost", "port": 5433, "dbname": "test", "user": "test"},
            connection_factory=Mock(return_value=connection),
        )

        self.assertEqual(version, PROJECT_SCOPE_SCHEMA_VERSION)
        connection.commit.assert_called_once()
        connection.rollback.assert_not_called()
        connection.close.assert_called_once()
        self.assertFalse(connection.autocommit)
        sql = "\n".join(statement for statement, _ in cursor.statements)
        self.assertIn(f"CREATE TABLE {ORGANIZATIONS_TABLE}", sql)
        self.assertIn(f"CREATE TABLE {PROJECTS_TABLE}", sql)
        self.assertIn(f"CREATE TABLE {PROVENANCE_TABLE}", sql)
        version_insert = next(
            (statement, parameters)
            for statement, parameters in cursor.statements
            if statement.strip().startswith(f"INSERT INTO {VERSION_TABLE}")
        )
        self.assertIn("%s", version_insert[0])
        self.assertEqual(version_insert[1], (PROJECT_SCOPE_SCHEMA_VERSION,))

    def test_legacy_migration_backfills_only_new_scope_columns(self) -> None:
        tables, columns, constraints = legacy_metadata()
        cursor = MigrationCursor(
            tables=tables,
            columns=columns,
            constraints=constraints,
            indexes=[],
            unassigned_count=0,
        )
        connection = self.make_connection(cursor)

        migrate_project_scope_schema(
            {"host": "localhost", "port": 5433, "dbname": "test", "user": "test"},
            connection_factory=Mock(return_value=connection),
        )

        update_statement, update_parameters = next(
            (statement, parameters)
            for statement, parameters in cursor.statements
            if statement.strip().startswith(f"UPDATE {PROVENANCE_TABLE}")
        )
        self.assertIn(
            "SET organization_id = %s, project_id = %s", update_statement
        )
        self.assertNotIn("SET run_id", update_statement)
        self.assertNotIn("image_sha256 =", update_statement)
        self.assertEqual(update_parameters, ("local", "default"))
        self.assertIn(
            f"CREATE INDEX {PROJECT_SCOPE_INDEX}",
            "\n".join(statement for statement, _ in cursor.statements),
        )
        connection.commit.assert_called_once()
        connection.rollback.assert_not_called()

    def test_current_version_is_idempotent_and_performs_no_ddl(self) -> None:
        tables, columns, constraints, indexes = current_metadata()
        cursor = MigrationCursor(
            tables=tables,
            columns=columns,
            constraints=constraints,
            indexes=indexes,
            version_rows=[(PROJECT_SCOPE_SCHEMA_VERSION,)],
            default_project_count=1,
        )
        connection = self.make_connection(cursor)

        version = migrate_project_scope_schema(
            {"host": "localhost", "port": 5433, "dbname": "test", "user": "test"},
            connection_factory=Mock(return_value=connection),
        )

        self.assertEqual(version, PROJECT_SCOPE_SCHEMA_VERSION)
        self.assertFalse(
            any(
                statement.strip().startswith(("CREATE", "ALTER", "UPDATE", "INSERT"))
                for statement, _ in cursor.statements
            )
        )
        constraints_query = next(
            statement
            for statement, _ in cursor.statements
            if "FROM pg_constraint" in statement
        )
        self.assertIn("source.oid = constraint_row.conrelid", constraints_query)
        self.assertNotIn("constraint_row.conname =", constraints_query)
        connection.commit.assert_called_once()
        connection.rollback.assert_not_called()

    def test_failure_rolls_back_schema_and_version_changes(self) -> None:
        cursor = MigrationCursor(
            tables=[],
            columns={},
            constraints=[],
            indexes=[],
            fail_on=f"CREATE TABLE {PROJECTS_TABLE}",
        )
        connection = self.make_connection(cursor)

        with self.assertRaisesRegex(Exception, "migration failed"):
            migrate_project_scope_schema(
                {"host": "localhost", "port": 5433, "dbname": "test", "user": "test"},
                connection_factory=Mock(return_value=connection),
            )

        connection.rollback.assert_called_once()
        connection.commit.assert_not_called()
        connection.close.assert_called_once()
        self.assertFalse(
            any(
                statement.strip().startswith(f"INSERT INTO {VERSION_TABLE}")
                for statement, _ in cursor.statements
            )
        )

    def test_legacy_backfill_validation_failure_rolls_back(self) -> None:
        tables, columns, constraints = legacy_metadata()
        cursor = MigrationCursor(
            tables=tables,
            columns=columns,
            constraints=constraints,
            indexes=[],
            unassigned_count=1,
        )
        connection = self.make_connection(cursor)

        with self.assertRaisesRegex(MigrationStateError, "unassigned rows"):
            migrate_project_scope_schema(
                {"host": "localhost", "port": 5433, "dbname": "test", "user": "test"},
                connection_factory=Mock(return_value=connection),
            )

        connection.rollback.assert_called_once()
        connection.commit.assert_not_called()

    def test_legacy_partial_scope_assignment_rolls_back(self) -> None:
        tables, columns, constraints = legacy_metadata()
        cursor = MigrationCursor(
            tables=tables,
            columns=columns,
            constraints=constraints,
            indexes=[],
            partial_scope_count=1,
        )
        connection = self.make_connection(cursor)

        with self.assertRaisesRegex(MigrationStateError, "unassigned rows"):
            migrate_project_scope_schema(
                {"host": "localhost", "port": 5433, "dbname": "test", "user": "test"},
                connection_factory=Mock(return_value=connection),
            )

        update = next(
            statement
            for statement, _ in cursor.statements
            if statement.strip().startswith(f"UPDATE {PROVENANCE_TABLE}")
        )
        validation = next(
            statement
            for statement, _ in cursor.statements
            if f"SELECT COUNT(*) FROM {PROVENANCE_TABLE}"
            in " ".join(statement.split())
        )
        self.assertIn(
            "WHERE organization_id IS NULL AND project_id IS NULL",
            " ".join(update.split()),
        )
        self.assertIn(
            "WHERE organization_id IS NULL OR project_id IS NULL",
            " ".join(validation.split()),
        )
        connection.rollback.assert_called_once()
        connection.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
