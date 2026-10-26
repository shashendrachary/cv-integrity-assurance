"""Command-line entry point for CV-Integrity-Assurance."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from src.demos import run_data_integrity_demo, run_model_tampering_demo
from src.integrity.data_checks import (
    ImageDependencyError,
    format_data_check,
    scan_dataset,
)
from src.anomaly.checks import load_inference_settings
from src.database.postgres import (
    DatabaseNotConfigured,
    DatabaseOperationError,
    initialize_provenance_table,
    load_database_settings,
)
from src.model.integrity import (
    ModelRegistryError,
    format_verification,
    register_model,
    verify_model,
)
from src.provenance.records import create_provenance_record, insert_provenance_record
from src.provenance.workflow import run_assessment
from src.project_context import ProjectContext, ProjectContextError, load_project_context
from src.reports.pdf_report import ReportGenerationError
from src.vision.inference import format_inference_result, run_inference


PROJECT_NAME = "CV-Integrity-Assurance"
PROJECT_TITLE = (
    "Trustworthy Computer Vision Integrity Assurance for Data, Models "
    "and Inference Outputs in Multi-Contributor Pipelines"
)
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "logs" / "assurance.log"

IMPLEMENTED_CAPABILITIES = (
    "Dataset integrity checks",
    "Model baseline registration and integrity verification",
    "Local YOLO object detection with image-quality review checks",
    "PostgreSQL inference provenance evidence",
    "End-to-end assessment with local PDF reports",
)

PLANNED_CAPABILITIES: tuple[str, ...] = ()


def configure_logging() -> logging.Logger:
    """Use file logging when possible, without blocking CLI startup if it fails."""
    logger = logging.getLogger("integrity_assurance")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    except OSError as error:
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
        logger.warning("File logging is unavailable; using stderr: %s", error)
    else:
        logger.addHandler(handler)

    for active_handler in logger.handlers:
        active_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
    return logger


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="assurance.py",
        description=f"Command-line foundation for {PROJECT_NAME}.",
    )
    commands = parser.add_subparsers(dest="command")

    def add_project_config_argument(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--project-config",
            type=Path,
            help=(
                "select a project configuration (relative paths use its directory; "
                "default: project config.yaml)"
            ),
        )

    data_parser = commands.add_parser(
        "check-data",
        help="validate dataset images and YOLO labels",
        description="Check image integrity, duplicates, YOLO labels, and file pairings.",
    )
    data_parser.add_argument(
        "--images-dir",
        type=Path,
        help="override the configured image directory",
    )
    data_parser.add_argument(
        "--labels-dir",
        type=Path,
        help="override the configured label directory",
    )
    add_project_config_argument(data_parser)
    register_parser = commands.add_parser(
        "register-model",
        help="register a model file's SHA-256 baseline",
        description="Create or explicitly replace a local model integrity baseline.",
    )
    register_parser.add_argument("--id", required=True, help="unique model identifier")
    register_parser.add_argument("--path", required=True, type=Path, help="model file path")
    register_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    register_parser.add_argument(
        "--replace",
        action="store_true",
        help="explicitly replace an existing trusted baseline",
    )
    add_project_config_argument(register_parser)
    verify_parser = commands.add_parser(
        "verify-model",
        help="verify a registered model file against its baseline",
        description="Compare the current model SHA-256 with its registered baseline.",
    )
    verify_parser.add_argument("--id", required=True, help="registered model identifier")
    verify_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(verify_parser)
    inference_parser = commands.add_parser(
        "infer",
        help="run integrity-gated local YOLO inference",
        description="Analyze an image and run CPU detection only with a verified model.",
    )
    inference_parser.add_argument("--id", required=True, help="registered model identifier")
    inference_parser.add_argument("--image", required=True, type=Path, help="input image path")
    inference_parser.add_argument(
        "--conf",
        type=float,
        help="confidence threshold from 0 to 1 (default: config.yaml setting)",
    )
    inference_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(inference_parser)
    commands.add_parser(
        "init-db",
        help="create the PostgreSQL inference provenance table if absent",
        description="Explicitly initialize the non-destructive provenance schema.",
    )
    assessment_parser = commands.add_parser(
        "assess",
        help="run inference, record available evidence, and generate a PDF report",
        description="Run a complete local image/model assurance assessment.",
    )
    assessment_parser.add_argument("--id", required=True, help="registered model identifier")
    assessment_parser.add_argument("--image", required=True, type=Path, help="input image path")
    assessment_parser.add_argument(
        "--conf",
        type=float,
        help="confidence threshold from 0 to 1 (default: config.yaml setting)",
    )
    assessment_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(assessment_parser)
    tamper_demo_parser = commands.add_parser(
        "demo-model-tampering",
        help="safely demonstrate model hash mismatch using a temporary copy",
    )
    tamper_demo_parser.add_argument(
        "--id",
        default="yolo11n",
        help="registered source model identifier (default: yolo11n)",
    )
    tamper_demo_parser.add_argument(
        "--registry",
        type=Path,
        help="override the registry path configured in config.yaml",
    )
    add_project_config_argument(tamper_demo_parser)
    commands.add_parser(
        "demo-data-integrity",
        help="demonstrate image and YOLO-label findings with temporary files",
    )
    return parser


def _project_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root unless already absolute."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_cli_project_context(project_config: Path | None) -> ProjectContext:
    configuration_path = (
        _project_path(project_config)
        if project_config is not None
        else PROJECT_ROOT / "config.yaml"
    )
    return load_project_context(configuration_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the foundation status, dataset check, or model-integrity command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    logger = configure_logging()
    project_context: ProjectContext | None = None
    if arguments.command in {
        "check-data",
        "register-model",
        "verify-model",
        "infer",
        "assess",
        "demo-model-tampering",
    }:
        try:
            project_context = _load_cli_project_context(arguments.project_config)
        except ProjectContextError as error:
            logger.error("Project configuration could not be loaded: %s", error)
            print(f"Project configuration failed: {error}", file=sys.stderr)
            return 2
        print(
            "Project context: "
            f"{project_context.organization_id}/{project_context.project_id}"
        )

    if arguments.command == "init-db":
        try:
            settings = load_database_settings(PROJECT_ROOT / ".env")
            if settings is None:
                raise DatabaseNotConfigured(
                    "PostgreSQL settings are not configured. Set POSTGRES_DB and "
                    "POSTGRES_USER in the environment or project .env file."
                )
            initialize_provenance_table(settings)
        except (DatabaseNotConfigured, DatabaseOperationError) as error:
            print(f"Database initialization failed: {error}", file=sys.stderr)
            return 2
        print("PostgreSQL provenance table initialized (existing records preserved).")
        return 0

    if arguments.command == "demo-data-integrity":
        run_data_integrity_demo()
        return 0

    if arguments.command == "demo-model-tampering":
        assert project_context is not None
        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            run_model_tampering_demo(
                arguments.id,
                registry_path,
                project_root=project_context.base_path,
            )
        except (OSError, ValueError, ModelRegistryError, RuntimeError) as error:
            print(f"Model tampering demo failed: {error}", file=sys.stderr)
            return 2
        return 0

    if arguments.command in {"register-model", "verify-model"}:
        assert project_context is not None
        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            if arguments.command == "register-model":
                registration = register_model(
                    arguments.id,
                    project_context.resolve_path(arguments.path),
                    registry_path,
                    replace=arguments.replace,
                )
                logger.info("Registered model baseline: %s", arguments.id)
                print(f"Model ID: {registration['model_id']}")
                print(f"File: {registration['file_path']}")
                print(f"Expected SHA-256: {registration['expected_hash']}")
                print(f"Registered at: {registration['registered_at']}")
                print(
                    "Note: registration establishes a local baseline; it does not "
                    "prove the original model is authentic."
                )
                return 0

            result = verify_model(
                arguments.id,
                registry_path,
                project_root=project_context.base_path,
            )
            print(format_verification(result))
            logger.info("Model verification %s: %s", result.status, arguments.id)
            return 0 if result.status == "VERIFIED" else 2
        except (OSError, ValueError, ModelRegistryError) as error:
            logger.error("Model operation could not be completed: %s", error)
            print(f"Model operation failed: {error}", file=sys.stderr)
            return 2

    if arguments.command == "infer":
        assert project_context is not None
        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            anomaly_thresholds, configured_confidence = load_inference_settings(
                project_context.configuration_path
            )
            result = run_inference(
                project_context.resolve_path(arguments.image),
                arguments.id,
                registry_path,
                confidence=(
                    arguments.conf
                    if arguments.conf is not None
                    else configured_confidence
                ),
                anomaly_thresholds=anomaly_thresholds,
                project_root=project_context.base_path,
            )
        except (OSError, ValueError, ModelRegistryError) as error:
            logger.error("Inference could not be started: %s", error)
            print(f"Inference failed: {error}", file=sys.stderr)
            return 2

        print(format_inference_result(result))
        record = create_provenance_record(
            project_context.resolve_path(arguments.image),
            result,
            {
                "confidence": result.confidence,
                "device": "cpu",
                "anomaly_thresholds": {
                    "minimum_width": anomaly_thresholds.minimum_width,
                    "minimum_height": anomaly_thresholds.minimum_height,
                    "maximum_width": anomaly_thresholds.maximum_width,
                    "maximum_height": anomaly_thresholds.maximum_height,
                    "minimum_aspect_ratio": anomaly_thresholds.minimum_aspect_ratio,
                    "maximum_aspect_ratio": anomaly_thresholds.maximum_aspect_ratio,
                    "minimum_mean_brightness": anomaly_thresholds.minimum_mean_brightness,
                    "maximum_mean_brightness": anomaly_thresholds.maximum_mean_brightness,
                    "minimum_laplacian_variance": anomaly_thresholds.minimum_laplacian_variance,
                },
            },
        )
        print(f"Provenance run ID: {record.run_id}")
        try:
            database_settings = load_database_settings(PROJECT_ROOT / ".env")
            if database_settings is None:
                print(
                    "Evidence persistence: skipped (PostgreSQL is not configured). "
                    "Set POSTGRES_DB and POSTGRES_USER or run init-db after setup."
                )
            else:
                insert_provenance_record(record, database_settings)
                print("Evidence persistence: saved to PostgreSQL.")
        except (DatabaseNotConfigured, DatabaseOperationError) as error:
            logger.warning("Provenance evidence was not persisted: %s", error)
            print(
                "Evidence persistence: not saved; PostgreSQL is unavailable or "
                f"misconfigured ({error})."
            )
        logger.info(
            "Inference run status %s for model %s",
            result.status,
            arguments.id,
        )
        return 0 if result.status == "VERIFIED" else 2

    if arguments.command == "assess":
        assert project_context is not None
        image_path = project_context.resolve_path(arguments.image)
        try:
            registry_path = (
                project_context.resolve_path(arguments.registry)
                if arguments.registry is not None
                else project_context.model_registry
            )
            report_directory = project_context.reports_directory
            anomaly_thresholds, configured_confidence = load_inference_settings(
                project_context.configuration_path
            )
            confidence = (
                arguments.conf
                if arguments.conf is not None
                else configured_confidence
            )
            database_note = None
            try:
                database_settings = load_database_settings(PROJECT_ROOT / ".env")
            except DatabaseNotConfigured as error:
                database_settings = None
                database_note = (
                    f"skipped (database configuration unavailable: {error})"
                )

            assessment = run_assessment(
                image_path,
                arguments.id,
                registry_path,
                report_directory,
                confidence=confidence,
                anomaly_thresholds=anomaly_thresholds,
                database_settings=database_settings,
                database_note=database_note,
                project_root=project_context.base_path,
            )
        except (
            OSError,
            ValueError,
            ModelRegistryError,
            ReportGenerationError,
        ) as error:
            logger.error("Assessment could not be completed: %s", error)
            print(f"Assessment failed: {error}", file=sys.stderr)
            return 2

        inference_summary = format_inference_result(assessment.inference).replace(
            "Run status:", "Inference status:", 1
        )
        print(inference_summary)
        print(f"Overall status: {assessment.overall_status}")
        print(f"Provenance run ID: {assessment.provenance.run_id}")
        print(f"PostgreSQL evidence: {assessment.persistence_status}")
        print(f"PDF report: {assessment.report_path}")
        logger.info(
            "Assessment %s completed with status %s",
            assessment.provenance.run_id,
            assessment.overall_status,
        )
        return 0 if assessment.overall_status == "VERIFIED" else 2

    if arguments.command == "check-data":
        assert project_context is not None
        try:
            result = scan_dataset(
                project_context.resolve_path(arguments.images_dir)
                if arguments.images_dir is not None
                else project_context.images_directory,
                project_context.resolve_path(arguments.labels_dir)
                if arguments.labels_dir is not None
                else project_context.labels_directory,
            )
        except (OSError, ValueError, ImageDependencyError) as error:
            logger.error("Data check could not be completed: %s", error)
            print(f"Data check failed: {error}", file=sys.stderr)
            return 2

        logger.info(
            "Data check completed: %d images, %d valid labels",
            result.images_examined,
            result.valid_label_files,
        )
        print(format_data_check(result))
        return 2 if result.operational_errors else 0

    logger.info("Foundation status requested")

    required_directories = (
        PROJECT_ROOT / "dataset" / "images",
        PROJECT_ROOT / "dataset" / "labels",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "reports",
        PROJECT_ROOT / "logs",
    )
    missing_directories = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in required_directories
        if not path.is_dir()
    ]

    print(f"Project: {PROJECT_NAME}")
    print(f"Title: {PROJECT_TITLE}")
    if missing_directories:
        print("Initialization status: incomplete")
        print(f"Missing directories: {', '.join(missing_directories)}")
    else:
        print("Initialization status: ready (foundation only)")

    print("Implemented capabilities:")
    for capability in IMPLEMENTED_CAPABILITIES:
        print(f"- {capability}")
    print("Planned capabilities (not implemented yet):")
    for capability in PLANNED_CAPABILITIES:
        print(f"- {capability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
