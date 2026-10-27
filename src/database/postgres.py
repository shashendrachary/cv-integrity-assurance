"""Small PostgreSQL connection and schema helpers for provenance evidence."""

from __future__ import annotations

import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Callable


class DatabaseNotConfigured(RuntimeError):
    """Raised when database settings are absent or incomplete."""


class DatabaseOperationError(RuntimeError):
    """Raised when PostgreSQL cannot complete a database operation."""


ConnectionSettings = dict[str, str | int]
ConnectionFactory = Callable[..., Any]

PROJECT_SCOPE_SCHEMA_VERSION = 1
MIGRATION_LOCK_ID = 731904218
VERSION_TABLE = "assurance_schema_version"
ORGANIZATIONS_TABLE = "organizations"
PROJECTS_TABLE = "projects"
PROVENANCE_TABLE = "inference_provenance"
PROJECT_SCOPE_FOREIGN_KEY = "inference_provenance_project_scope_fkey"
PROJECT_SCOPE_INDEX = "inference_provenance_project_recorded_at_idx"

LEGACY_PROVENANCE_COLUMNS: dict[str, tuple[str, bool]] = {
    "run_id": ("uuid", True),
    "recorded_at": ("timestamptz", True),
    "image_path": ("text", True),
    "image_sha256": ("bpchar", False),
    "model_id": ("text", True),
    "model_path": ("text", False),
    "model_expected_sha256": ("bpchar", False),
    "model_actual_sha256": ("bpchar", False),
    "model_verification_status": ("text", True),
    "confidence": ("float8", False),
    "detections": ("jsonb", True),
    "image_metrics": ("jsonb", True),
    "configuration": ("jsonb", True),
    "run_status": ("text", True),
    "inference_succeeded": ("bool", True),
    "reasons": ("jsonb", True),
}


@dataclass(frozen=True)
class SchemaConstraint:
    """Relevant PostgreSQL constraint metadata used for safe state detection."""

    table: str
    name: str
    kind: str
    columns: tuple[str, ...]
    referenced_table: str | None = None
    referenced_columns: tuple[str, ...] = ()
    definition: str | None = None
    validated: bool = True


@dataclass(frozen=True)
class SchemaIndex:
    """Catalog metadata required to validate the scoped provenance index."""

    access_method: str
    valid: bool
    ready: bool
    unique: bool
    predicate_free: bool
    key_columns: tuple[str | None, ...]
    total_columns: int
    plain_ascending: bool
    default_operator_classes: bool


@dataclass(frozen=True)
class ProjectScopeSchema:
    """Database metadata needed to choose a safe project-scope migration plan."""

    tables: frozenset[str]
    columns: dict[str, dict[str, tuple[str, bool]]]
    constraints: tuple[SchemaConstraint, ...]
    indexes: dict[str, SchemaIndex]
    version: int | None
    version_table_exists: bool
    non_table_objects: frozenset[str] = frozenset()
    default_project_exists: bool = False


class MigrationStateError(DatabaseOperationError):
    """Raised when the existing database schema cannot be migrated safely."""


def _read_dotenv(env_path: Path) -> dict[str, str]:
    """Read simple KEY=value lines without overriding process environment values."""
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatabaseNotConfigured(
            f"Cannot read {env_path}; check its permissions."
        ) from error
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].strip()
        key, separator, value = stripped.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_database_settings(
    env_path: Path,
    environment: dict[str, str] | None = None,
) -> ConnectionSettings | None:
    """Load settings from environment first, then .env; return None if unconfigured."""
    file_values = _read_dotenv(env_path)
    effective = dict(file_values)
    effective.update(os.environ if environment is None else environment)

    keys = (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    )
    settings = {key: effective[key].strip() for key in keys if effective.get(key, "").strip()}
    if not settings:
        return None

    required = ("POSTGRES_DB", "POSTGRES_USER")
    missing = [key for key in required if not settings.get(key)]
    placeholders = {"your_database_name", "your_username", "your_password_here"}
    if missing or any(value.casefold() in placeholders for value in settings.values()):
        raise DatabaseNotConfigured(
            "PostgreSQL settings are incomplete or still placeholders. Configure "
            "POSTGRES_DB and POSTGRES_USER (and POSTGRES_PASSWORD if required) "
            "in the environment or project .env file."
        )
    try:
        port = int(settings.get("POSTGRES_PORT", "5433"))
    except ValueError as error:
        raise DatabaseNotConfigured("POSTGRES_PORT must be an integer.") from error
    if not 1 <= port <= 65535:
        raise DatabaseNotConfigured("POSTGRES_PORT must be between 1 and 65535.")

    connection: ConnectionSettings = {
        "host": settings.get("POSTGRES_HOST", "localhost"),
        "port": port,
        "dbname": settings["POSTGRES_DB"],
        "user": settings["POSTGRES_USER"],
        "connect_timeout": 5,
    }
    if settings.get("POSTGRES_PASSWORD"):
        connection["password"] = settings["POSTGRES_PASSWORD"]
    return connection


def _connect(settings: ConnectionSettings, connection_factory: ConnectionFactory | None) -> Any:
    if connection_factory is not None:
        try:
            return connection_factory(**settings)
        except Exception as error:
            raise DatabaseOperationError(
                f"Could not connect to PostgreSQL at {settings['host']}:{settings['port']}. "
                "Check that PostgreSQL is running and the configured database/user are valid."
            ) from error
    try:
        import psycopg
    except ImportError as error:
        raise DatabaseOperationError(
            "PostgreSQL support requires Psycopg 3. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error
    try:
        return psycopg.connect(**settings)
    except Exception as error:
        raise DatabaseOperationError(
            f"Could not connect to PostgreSQL at {settings['host']}:{settings['port']}. "
            "Check that PostgreSQL is running and the configured database/user are valid."
        ) from error


def run_statement(
    settings: ConnectionSettings,
    statement: str,
    parameters: tuple[Any, ...] | None = None,
    connection_factory: ConnectionFactory | None = None,
) -> None:
    run_transaction(
        settings,
        ((statement, parameters),),
        connection_factory=connection_factory,
    )


def run_transaction(
    settings: ConnectionSettings,
    statements: tuple[tuple[str, tuple[Any, ...] | None], ...],
    connection_factory: ConnectionFactory | None = None,
) -> None:
    """Execute parameterized statements atomically on one managed connection."""
    connection = _connect(settings, connection_factory)
    try:
        cursor = connection.cursor()
        try:
            for statement, parameters in statements:
                if parameters is not None:
                    cursor.execute(statement, parameters)
                else:
                    cursor.execute(statement)
            connection.commit()
        finally:
            cursor.close()
    except Exception as error:
        try:
            connection.rollback()
        except Exception:
            pass
        if isinstance(error, DatabaseOperationError):
            raise
        raise DatabaseOperationError(
            "PostgreSQL operation failed; the transaction was rolled back. "
            "Check the database schema and permissions. If project-scoped provenance "
            "is being written to a legacy schema, explicitly run "
            "'python assurance.py migrate-db' after backing up the database."
        ) from error
    finally:
        connection.close()


def initialize_provenance_table(
    settings: ConnectionSettings,
    connection_factory: ConnectionFactory | None = None,
) -> None:
    """Create the provenance table if absent; never alter or delete existing data."""
    statement = """
        CREATE TABLE IF NOT EXISTS inference_provenance (
            run_id UUID PRIMARY KEY,
            recorded_at TIMESTAMPTZ NOT NULL,
            image_path TEXT NOT NULL,
            image_sha256 CHAR(64),
            model_id TEXT NOT NULL,
            model_path TEXT,
            model_expected_sha256 CHAR(64),
            model_actual_sha256 CHAR(64),
            model_verification_status TEXT NOT NULL,
            confidence DOUBLE PRECISION,
            detections JSONB NOT NULL,
            image_metrics JSONB NOT NULL,
            configuration JSONB NOT NULL,
            run_status TEXT NOT NULL,
            inference_succeeded BOOLEAN NOT NULL,
            reasons JSONB NOT NULL
        )
    """
    run_statement(statement=statement, settings=settings, connection_factory=connection_factory)


def _inspect_project_scope_schema(cursor: Any) -> ProjectScopeSchema:
    target_tables = (
        VERSION_TABLE,
        ORGANIZATIONS_TABLE,
        PROJECTS_TABLE,
        PROVENANCE_TABLE,
    )
    cursor.execute(
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = current_schema()
          AND table_name IN (%s, %s, %s, %s)
        """,
        target_tables,
    )
    table_rows = cursor.fetchall()
    tables = frozenset(
        name for name, table_type in table_rows if table_type == "BASE TABLE"
    )
    non_table_objects = frozenset(
        name for name, table_type in table_rows if table_type != "BASE TABLE"
    )

    columns: dict[str, dict[str, tuple[str, bool]]] = {}
    for table in target_tables:
        if table not in tables:
            continue
        cursor.execute(
            """
            SELECT column_name, udt_name, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = %s
            """,
            (table,),
        )
        columns[table] = {
            name: (udt_name, is_nullable == "NO")
            for name, udt_name, is_nullable in cursor.fetchall()
        }

    constraints: list[SchemaConstraint] = []
    if tables:
        cursor.execute(
            """
            SELECT
                source.relname,
                constraint_row.conname,
                constraint_row.contype,
                source_column.attname,
                CASE
                    WHEN referenced_schema.nspname = source_schema.nspname
                    THEN referenced.relname
                    ELSE referenced_schema.nspname || '.' || referenced.relname
                END,
                referenced_column.attname,
                pg_get_constraintdef(constraint_row.oid, TRUE),
                constraint_row.convalidated
            FROM pg_constraint AS constraint_row
            JOIN pg_class AS source
              ON source.oid = constraint_row.conrelid
            JOIN pg_namespace AS source_schema
              ON source_schema.oid = source.relnamespace
            LEFT JOIN pg_class AS referenced
              ON referenced.oid = constraint_row.confrelid
            LEFT JOIN pg_namespace AS referenced_schema
              ON referenced_schema.oid = referenced.relnamespace
            LEFT JOIN LATERAL unnest(
                CASE
                    WHEN constraint_row.contype IN ('p', 'f', 'u', 'x')
                    THEN constraint_row.conkey
                END
            )
              WITH ORDINALITY AS source_key(attnum, position)
              ON TRUE
            LEFT JOIN pg_attribute AS source_column
              ON source_column.attrelid = source.oid
             AND source_column.attnum = source_key.attnum
            LEFT JOIN LATERAL unnest(
                CASE WHEN constraint_row.contype = 'f' THEN constraint_row.confkey END
            )
              WITH ORDINALITY AS referenced_key(attnum, position)
              ON referenced_key.position = source_key.position
            LEFT JOIN pg_attribute AS referenced_column
              ON referenced_column.attrelid = referenced.oid
             AND referenced_column.attnum = referenced_key.attnum
            WHERE source_schema.nspname = current_schema()
              AND source.relname IN (%s, %s, %s, %s)
            ORDER BY source.relname, constraint_row.conname, source_key.position
            """,
            target_tables,
        )
        grouped: dict[
            tuple[str, str, str],
            tuple[list[str], str | None, list[str], str, bool],
        ] = {}
        constraint_kinds = {
            "p": "PRIMARY KEY",
            "f": "FOREIGN KEY",
            "c": "CHECK",
            "u": "UNIQUE",
            "x": "EXCLUSION",
        }
        for (
            table,
            name,
            constraint_type,
            column,
            ref_table,
            ref_column,
            definition,
            validated,
        ) in cursor.fetchall():
            kind = constraint_kinds.get(constraint_type, str(constraint_type))
            key = (table, name, kind)
            if key not in grouped:
                grouped[key] = ([], ref_table, [], definition, validated)
            columns_for_constraint, referenced_table, referenced_columns, _, _ = grouped[key]
            if column is not None:
                columns_for_constraint.append(column)
            if ref_column is not None:
                referenced_columns.append(ref_column)
            if referenced_table is None and ref_table is not None:
                grouped[key] = (
                    columns_for_constraint,
                    ref_table,
                    referenced_columns,
                    definition,
                    validated,
                )
        constraints = [
            SchemaConstraint(
                table,
                name,
                kind,
                tuple(constraint_columns),
                referenced_table,
                tuple(referenced_columns),
                definition,
                validated,
            )
            for (table, name, kind), (
                constraint_columns,
                referenced_table,
                referenced_columns,
                definition,
                validated,
            ) in grouped.items()
        ]

    indexes: dict[str, SchemaIndex] = {}
    if PROVENANCE_TABLE in tables:
        cursor.execute(
            """
            SELECT
                index_relation.relname,
                access_method.amname,
                index_row.indisvalid,
                index_row.indisready,
                index_row.indisunique,
                (index_row.indpred IS NULL),
                index_row.indnkeyatts,
                index_row.indnatts,
                array_agg(index_column.attname ORDER BY index_key.position)
                    FILTER (WHERE index_key.position <= index_row.indnkeyatts),
                bool_and(index_option.option = 0),
                bool_and(index_operator_class.opcdefault)
            FROM pg_index AS index_row
            JOIN pg_class AS table_relation
              ON table_relation.oid = index_row.indrelid
            JOIN pg_namespace AS table_schema
              ON table_schema.oid = table_relation.relnamespace
            JOIN pg_class AS index_relation
              ON index_relation.oid = index_row.indexrelid
            JOIN pg_am AS access_method
              ON access_method.oid = index_relation.relam
            LEFT JOIN LATERAL unnest(index_row.indkey)
              WITH ORDINALITY AS index_key(attnum, position)
              ON index_key.position <= index_row.indnkeyatts
            LEFT JOIN pg_attribute AS index_column
              ON index_column.attrelid = table_relation.oid
             AND index_column.attnum = index_key.attnum
            LEFT JOIN LATERAL unnest(index_row.indoption)
              WITH ORDINALITY AS index_option(option, position)
              ON index_option.position = index_key.position
            LEFT JOIN LATERAL unnest(index_row.indclass)
              WITH ORDINALITY AS index_operator_class_key(opclass_oid, position)
              ON index_operator_class_key.position = index_key.position
            LEFT JOIN pg_opclass AS index_operator_class
              ON index_operator_class.oid = index_operator_class_key.opclass_oid
            WHERE table_schema.nspname = current_schema()
              AND table_relation.relname = %s
            GROUP BY
                index_relation.relname,
                access_method.amname,
                index_row.indisvalid,
                index_row.indisready,
                index_row.indisunique,
                (index_row.indpred IS NULL),
                index_row.indnkeyatts,
                index_row.indnatts
            """,
            (PROVENANCE_TABLE,),
        )
        indexes = {
            name: SchemaIndex(
                access_method,
                valid,
                ready,
                unique,
                predicate_free,
                tuple(key_columns or ()),
                total_columns,
                plain_ascending,
                default_operator_classes,
            )
            for (
                name,
                access_method,
                valid,
                ready,
                unique,
                predicate_free,
                _,
                total_columns,
                key_columns,
                plain_ascending,
                default_operator_classes,
            ) in cursor.fetchall()
        }

    version: int | None = None
    if VERSION_TABLE in tables:
        version_columns = columns.get(VERSION_TABLE, {})
        if version_columns != {
            "singleton": ("bool", True),
            "version": ("int4", True),
        }:
            raise MigrationStateError(
                "Project-scope schema version table has an unexpected shape."
            )
        cursor.execute(
            f"SELECT version FROM {VERSION_TABLE} WHERE singleton = TRUE"
        )
        versions = cursor.fetchall()
        if len(versions) != 1 or not isinstance(versions[0][0], int):
            raise MigrationStateError(
                "Project-scope schema version table is empty or ambiguous."
            )
        version = versions[0][0]

    default_project_exists = False
    if PROJECTS_TABLE in tables:
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {PROJECTS_TABLE}
            WHERE organization_id = %s AND project_id = %s
            """,
            ("local", "default"),
        )
        default_project_exists = cursor.fetchone()[0] == 1

    return ProjectScopeSchema(
        tables,
        columns,
        tuple(constraints),
        indexes,
        version,
        VERSION_TABLE in tables,
        non_table_objects,
        default_project_exists,
    )


def _constraint_matches(
    schema: ProjectScopeSchema,
    *,
    table: str,
    name: str,
    kind: str,
    columns: tuple[str, ...],
    referenced_table: str | None = None,
    referenced_columns: tuple[str, ...] = (),
    definition: str,
) -> bool:
    expected_definition = _normalize_constraint_definition(definition)
    return any(
        constraint.table == table
        and constraint.name == name
        and constraint.kind == kind
        and constraint.columns == columns
        and constraint.referenced_table == referenced_table
        and constraint.referenced_columns == referenced_columns
        and constraint.validated
        and constraint.definition is not None
        and _normalize_constraint_definition(constraint.definition)
        == expected_definition
        for constraint in schema.constraints
    )


def _normalize_constraint_definition(definition: str) -> str:
    without_quotes = re.sub(r'"([^"]+)"', r"\1", definition)
    return " ".join(without_quotes.casefold().split())


def _validate_legacy_provenance_schema(schema: ProjectScopeSchema) -> None:
    columns = schema.columns.get(PROVENANCE_TABLE, {})
    for name, (expected_type, required) in LEGACY_PROVENANCE_COLUMNS.items():
        observed = columns.get(name)
        if observed is None or observed[0] != expected_type or (
            required and not observed[1]
        ):
            raise MigrationStateError(
                "Existing inference_provenance schema conflicts with the "
                f"supported legacy column {name!r}."
            )
    if not _constraint_matches(
        schema,
        table=PROVENANCE_TABLE,
        name="inference_provenance_pkey",
        kind="PRIMARY KEY",
        columns=("run_id",),
        definition="PRIMARY KEY (run_id)",
    ):
        raise MigrationStateError(
            "Existing inference_provenance table has an unexpected primary key."
        )


def _validate_current_project_schema(schema: ProjectScopeSchema) -> None:
    if schema.non_table_objects:
        raise MigrationStateError(
            "Project-scope schema object name conflicts with a non-table object."
        )
    if not {
        VERSION_TABLE,
        ORGANIZATIONS_TABLE,
        PROJECTS_TABLE,
        PROVENANCE_TABLE,
    }.issubset(schema.tables):
        raise MigrationStateError(
            "Recorded project-scope schema version does not match existing tables."
        )

    _validate_legacy_provenance_schema(schema)
    expected_columns = {
        ORGANIZATIONS_TABLE: {"organization_id": ("text", True)},
        PROJECTS_TABLE: {
            "organization_id": ("text", True),
            "project_id": ("text", True),
        },
        VERSION_TABLE: {
            "singleton": ("bool", True),
            "version": ("int4", True),
        },
    }
    for table, required_columns in expected_columns.items():
        columns = schema.columns.get(table, {})
        if any(columns.get(name) != value for name, value in required_columns.items()):
            raise MigrationStateError(
                f"Recorded project-scope schema has conflicting {table} columns."
            )
    provenance_columns = schema.columns[PROVENANCE_TABLE]
    if provenance_columns.get("organization_id") != ("text", True) or (
        provenance_columns.get("project_id") != ("text", True)
    ):
        raise MigrationStateError(
            "Recorded project-scope schema has conflicting provenance scope columns."
        )

    required_constraints = (
        (
            VERSION_TABLE,
            "assurance_schema_version_pkey",
            "PRIMARY KEY",
            ("singleton",),
            None,
            (),
            "PRIMARY KEY (singleton)",
        ),
        (
            VERSION_TABLE,
            "assurance_schema_version_singleton_check",
            "CHECK",
            (),
            None,
            (),
            "CHECK (singleton)",
        ),
        (
            ORGANIZATIONS_TABLE,
            "organizations_pkey",
            "PRIMARY KEY",
            ("organization_id",),
            None,
            (),
            "PRIMARY KEY (organization_id)",
        ),
        (
            PROJECTS_TABLE,
            "projects_pkey",
            "PRIMARY KEY",
            ("organization_id", "project_id"),
            None,
            (),
            "PRIMARY KEY (organization_id, project_id)",
        ),
        (
            PROJECTS_TABLE,
            "projects_organization_id_fkey",
            "FOREIGN KEY",
            ("organization_id",),
            ORGANIZATIONS_TABLE,
            ("organization_id",),
            "FOREIGN KEY (organization_id) REFERENCES organizations(organization_id)",
        ),
        (
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
    if any(
        not _constraint_matches(
            schema,
            table=table,
            name=name,
            kind=kind,
            columns=columns,
            referenced_table=referenced_table,
            referenced_columns=referenced_columns,
            definition=definition,
        )
        for (
            table,
            name,
            kind,
            columns,
            referenced_table,
            referenced_columns,
            definition,
        ) in required_constraints
    ):
        raise MigrationStateError(
            "Recorded project-scope schema is missing or conflicts with required constraints."
        )

    scoped_index = schema.indexes.get(PROJECT_SCOPE_INDEX)
    if (
        scoped_index is None
        or scoped_index.access_method != "btree"
        or not scoped_index.valid
        or not scoped_index.ready
        or scoped_index.unique
        or not scoped_index.predicate_free
        or not scoped_index.plain_ascending
        or not scoped_index.default_operator_classes
        or scoped_index.key_columns
        != ("organization_id", "project_id", "recorded_at")
        or scoped_index.total_columns != 3
    ):
        raise MigrationStateError(
            "Recorded project-scope schema is missing its scoped provenance index."
        )
    if not schema.default_project_exists:
        raise MigrationStateError(
            "Recorded project-scope schema is missing the local/default project."
        )


def _plan_project_scope_migration(schema: ProjectScopeSchema) -> str:
    """Return new, legacy, or current; reject ambiguous/unsupported schema states."""
    if schema.non_table_objects:
        raise MigrationStateError(
            "A project-scope schema name conflicts with a non-table object."
        )
    if schema.version_table_exists:
        if schema.version is None:
            raise MigrationStateError(
                "Project-scope schema version is unavailable."
            )
        if schema.version > PROJECT_SCOPE_SCHEMA_VERSION:
            raise MigrationStateError(
                f"Database schema version {schema.version} is newer than supported "
                f"version {PROJECT_SCOPE_SCHEMA_VERSION}."
            )
        if schema.version != PROJECT_SCOPE_SCHEMA_VERSION:
            raise MigrationStateError(
                f"Unsupported project-scope schema version {schema.version}."
            )
        _validate_current_project_schema(schema)
        return "current"

    project_tables = {ORGANIZATIONS_TABLE, PROJECTS_TABLE} & schema.tables
    if project_tables:
        raise MigrationStateError(
            "Organization/project tables exist without a schema version; "
            "manual review is required."
        )

    if PROVENANCE_TABLE not in schema.tables:
        if schema.tables:
            raise MigrationStateError(
                "Unrecognized schema objects are present; refusing to initialize "
                "project scope automatically."
            )
        return "new"

    _validate_legacy_provenance_schema(schema)
    provenance_columns = schema.columns[PROVENANCE_TABLE]
    scope_columns = {"organization_id", "project_id"} & provenance_columns.keys()
    if scope_columns:
        raise MigrationStateError(
            "Project scope columns exist without a schema version; "
            "manual review is required."
        )
    return "legacy"


def migrate_project_scope_schema(
    settings: ConnectionSettings,
    connection_factory: ConnectionFactory | None = None,
) -> int:
    """Transactionally create or upgrade the project-scoped provenance schema."""
    connection = _connect(settings, connection_factory)
    try:
        connection.autocommit = False
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
            schema = _inspect_project_scope_schema(cursor)
            plan = _plan_project_scope_migration(schema)
            if plan == "current":
                connection.commit()
                return PROJECT_SCOPE_SCHEMA_VERSION

            cursor.execute(
                f"""
                CREATE TABLE {VERSION_TABLE} (
                    singleton BOOLEAN NOT NULL DEFAULT TRUE,
                    version INTEGER NOT NULL,
                    CONSTRAINT assurance_schema_version_pkey PRIMARY KEY (singleton),
                    CONSTRAINT assurance_schema_version_singleton_check CHECK (singleton)
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE {ORGANIZATIONS_TABLE} (
                    organization_id TEXT PRIMARY KEY
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE {PROJECTS_TABLE} (
                    organization_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    CONSTRAINT projects_pkey
                        PRIMARY KEY (organization_id, project_id),
                    CONSTRAINT projects_organization_id_fkey
                        FOREIGN KEY (organization_id)
                        REFERENCES {ORGANIZATIONS_TABLE} (organization_id)
                )
                """
            )
            cursor.execute(
                f"INSERT INTO {ORGANIZATIONS_TABLE} (organization_id) VALUES (%s)",
                ("local",),
            )
            cursor.execute(
                f"""
                INSERT INTO {PROJECTS_TABLE} (organization_id, project_id)
                VALUES (%s, %s)
                """,
                ("local", "default"),
            )

            if plan == "new":
                cursor.execute(
                    f"""
                    CREATE TABLE {PROVENANCE_TABLE} (
                        run_id UUID PRIMARY KEY,
                        recorded_at TIMESTAMPTZ NOT NULL,
                        image_path TEXT NOT NULL,
                        image_sha256 CHAR(64),
                        model_id TEXT NOT NULL,
                        model_path TEXT,
                        model_expected_sha256 CHAR(64),
                        model_actual_sha256 CHAR(64),
                        model_verification_status TEXT NOT NULL,
                        confidence DOUBLE PRECISION,
                        detections JSONB NOT NULL,
                        image_metrics JSONB NOT NULL,
                        configuration JSONB NOT NULL,
                        run_status TEXT NOT NULL,
                        inference_succeeded BOOLEAN NOT NULL,
                        reasons JSONB NOT NULL,
                        organization_id TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        CONSTRAINT {PROJECT_SCOPE_FOREIGN_KEY}
                            FOREIGN KEY (organization_id, project_id)
                            REFERENCES {PROJECTS_TABLE}
                                (organization_id, project_id)
                    )
                    """
                )
            else:
                cursor.execute(
                    f"""
                    ALTER TABLE {PROVENANCE_TABLE}
                    ADD COLUMN organization_id TEXT,
                    ADD COLUMN project_id TEXT
                    """
                )
                cursor.execute(
                    f"""
                    UPDATE {PROVENANCE_TABLE}
                    SET organization_id = %s, project_id = %s
                    WHERE organization_id IS NULL AND project_id IS NULL
                    """,
                    ("local", "default"),
                )
                cursor.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {PROVENANCE_TABLE}
                    WHERE organization_id IS NULL OR project_id IS NULL
                    """
                )
                unassigned_rows = cursor.fetchone()[0]
                if unassigned_rows:
                    raise MigrationStateError(
                        "Legacy provenance backfill left unassigned rows; "
                        "migration was rolled back."
                    )
                cursor.execute(
                    f"""
                    ALTER TABLE {PROVENANCE_TABLE}
                    ALTER COLUMN organization_id SET NOT NULL,
                    ALTER COLUMN project_id SET NOT NULL,
                    ADD CONSTRAINT {PROJECT_SCOPE_FOREIGN_KEY}
                        FOREIGN KEY (organization_id, project_id)
                        REFERENCES {PROJECTS_TABLE}
                            (organization_id, project_id)
                    """
                )

            cursor.execute(
                f"""
                CREATE INDEX {PROJECT_SCOPE_INDEX}
                ON {PROVENANCE_TABLE} (organization_id, project_id, recorded_at)
                """
            )
            cursor.execute(
                f"""
                INSERT INTO {VERSION_TABLE} (singleton, version)
                VALUES (TRUE, %s)
                """,
                (PROJECT_SCOPE_SCHEMA_VERSION,),
            )
            connection.commit()
            return PROJECT_SCOPE_SCHEMA_VERSION
        finally:
            cursor.close()
    except Exception as error:
        try:
            connection.rollback()
        except Exception:
            pass
        if isinstance(error, DatabaseOperationError):
            raise
        raise DatabaseOperationError(
            "Project-scope database migration failed; all migration changes "
            "were rolled back. Inspect the schema and migration logs before retrying."
        ) from error
    finally:
        connection.close()
