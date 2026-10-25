"""Small PostgreSQL connection and schema helpers for provenance evidence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


class DatabaseNotConfigured(RuntimeError):
    """Raised when database settings are absent or incomplete."""


class DatabaseOperationError(RuntimeError):
    """Raised when PostgreSQL cannot complete a database operation."""


ConnectionSettings = dict[str, str | int]
ConnectionFactory = Callable[..., Any]


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
    connection = _connect(settings, connection_factory)
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(statement, parameters) if parameters is not None else cursor.execute(statement)
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
            "Run 'python assurance.py init-db' and check database permissions."
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
