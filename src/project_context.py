"""Validated project identity and resource paths for CLI operations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PROJECT_SECTION_KEYS = {"organization_id", "project_id"}
PATH_SECTION_KEYS = {"images", "labels", "model_registry", "reports"}


class ProjectContextError(ValueError):
    """Raised when project identity or required resource paths are invalid."""


@dataclass(frozen=True)
class ProjectContext:
    """Project identity and paths resolved from one configuration file."""

    organization_id: str
    project_id: str
    configuration_path: Path
    base_path: Path
    images_directory: Path
    labels_directory: Path
    model_registry: Path
    image_registry: Path
    reports_directory: Path

    def resolve_path(self, path: Path) -> Path:
        """Resolve a CLI path relative to this configuration's directory."""
        return path if path.is_absolute() else (self.base_path / path).resolve()


def _parse_sections(configuration_path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {"project": {}, "paths": {}}
    try:
        lines = configuration_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProjectContextError(
            f"Cannot read project configuration {configuration_path}: {error}"
        ) from error
    except UnicodeError as error:
        raise ProjectContextError(
            f"Project configuration {configuration_path} is not valid UTF-8."
        ) from error

    current_section: str | None = None
    seen_sections: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        content = line.split("#", maxsplit=1)[0].rstrip()
        if not content.strip():
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise ProjectContextError(
                f"Project configuration line {line_number} uses tab indentation."
            )
        if not content[0].isspace():
            section_name, separator, value = content.partition(":")
            section_name = section_name.strip()
            if not separator or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_-]*", section_name
            ):
                raise ProjectContextError(
                    f"Invalid project configuration section on line {line_number}."
                )
            if value.strip():
                if section_name in sections:
                    raise ProjectContextError(
                        f"Project configuration line {line_number} must define "
                        f"{section_name}: as a mapping."
                    )
                current_section = None
                continue
            if section_name in sections and section_name in seen_sections:
                raise ProjectContextError(
                    f"Duplicate {section_name} section on configuration "
                    f"line {line_number}."
                )
            if section_name in sections:
                seen_sections.add(section_name)
            current_section = section_name
            continue
        if current_section not in sections:
            continue

        key, separator, value = content.strip().partition(":")
        if not separator or not key.strip() or not value.strip():
            raise ProjectContextError(
                f"Invalid {current_section} setting on configuration line "
                f"{line_number}; expected key: value."
            )
        key = key.strip()
        if key in sections[current_section]:
            raise ProjectContextError(
                f"Duplicate {current_section}.{key} setting on configuration "
                f"line {line_number}."
            )
        value = value.strip()
        if (
            value.startswith(("'", '"'))
            and (len(value) < 2 or value[-1] != value[0])
        ) or (
            value.endswith(("'", '"'))
            and not value.startswith(("'", '"'))
        ):
            raise ProjectContextError(
                f"Unmatched quote in {current_section}.{key} on configuration "
                f"line {line_number}."
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        sections[current_section][key] = value

    return sections


def load_project_context(configuration_path: Path) -> ProjectContext:
    """Load required project identity and resource paths from simple YAML mappings.

    Relative resource paths use the selected configuration file's parent
    directory as their base. This intentionally supports the project's current
    simple YAML shape without adding a runtime YAML dependency.
    """
    configuration_path = configuration_path.resolve()
    sections = _parse_sections(configuration_path)
    project = sections["project"]
    paths = sections["paths"]

    for key in sorted(PROJECT_SECTION_KEYS):
        value = project.get(key)
        if not value:
            raise ProjectContextError(
                f"Project configuration {configuration_path} must define "
                f"project.{key}."
            )
        if not IDENTIFIER_PATTERN.fullmatch(value):
            raise ProjectContextError(
                f"Invalid project identifier project.{key}: expected 1-64 "
                "letters, digits, dots, underscores, or hyphens, starting "
                "with a letter or digit."
            )

    missing_paths = sorted(key for key in PATH_SECTION_KEYS if not paths.get(key))
    if missing_paths:
        raise ProjectContextError(
            f"Project configuration {configuration_path} must define "
            f"paths.{', paths.'.join(missing_paths)}."
        )

    base_path = configuration_path.parent

    def resolve_resource(key: str) -> Path:
        path = Path(paths[key])
        return path if path.is_absolute() else (base_path / path).resolve()

    return ProjectContext(
        organization_id=project["organization_id"],
        project_id=project["project_id"],
        configuration_path=configuration_path,
        base_path=base_path,
        images_directory=resolve_resource("images"),
        labels_directory=resolve_resource("labels"),
        model_registry=resolve_resource("model_registry"),
        image_registry=(
            resolve_resource("image_registry")
            if paths.get("image_registry")
            else (base_path / "models" / "image_registry.json").resolve()
        ),
        reports_directory=resolve_resource("reports"),
    )
