"""Tests for project identity and resource-path configuration."""

import tempfile
import unittest
from pathlib import Path

from assurance import build_parser
from src.project_context import ProjectContextError, load_project_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_config(self, content: str, path: Path | None = None) -> Path:
        config_path = path or self.root / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(content, encoding="utf-8")
        return config_path

    def test_default_configuration_uses_stable_legacy_identity(self) -> None:
        context = load_project_context(PROJECT_ROOT / "config.yaml")

        self.assertEqual(context.organization_id, "local")
        self.assertEqual(context.project_id, "default")
        self.assertEqual(context.configuration_path, PROJECT_ROOT / "config.yaml")
        self.assertEqual(context.images_directory, PROJECT_ROOT / "dataset" / "images")
        self.assertEqual(context.labels_directory, PROJECT_ROOT / "dataset" / "labels")
        self.assertEqual(
            context.model_registry,
            PROJECT_ROOT / "models" / "model_registry.json",
        )
        self.assertEqual(context.reports_directory, PROJECT_ROOT / "reports")

    def test_alternative_configuration_loads_identity_and_relative_paths(self) -> None:
        configuration = self.write_config(
            "project:\n"
            "  organization_id: org-blue\n"
            "  project_id: road-inspection\n"
            "paths:\n"
            "  images: input/images\n"
            "  labels: input/labels\n"
            "  model_registry: models/registry.json\n"
            "  reports: output/reports\n",
            self.root / "projects" / "road" / "project.yaml",
        )

        context = load_project_context(configuration)
        base = configuration.parent

        self.assertEqual(context.organization_id, "org-blue")
        self.assertEqual(context.project_id, "road-inspection")
        self.assertEqual(context.base_path, base)
        self.assertEqual(context.images_directory, base / "input" / "images")
        self.assertEqual(context.labels_directory, base / "input" / "labels")
        self.assertEqual(
            context.model_registry, base / "models" / "registry.json"
        )
        self.assertEqual(context.reports_directory, base / "output" / "reports")

    def test_absolute_resource_paths_remain_absolute(self) -> None:
        absolute_images = self.root / "external-images"
        absolute_labels = self.root / "external-labels"
        absolute_registry = self.root / "external-registry.json"
        absolute_reports = self.root / "external-reports"
        configuration = self.write_config(
            "project:\n"
            "  organization_id: org\n"
            "  project_id: project\n"
            "paths:\n"
            f"  images: {absolute_images}\n"
            f"  labels: {absolute_labels}\n"
            f"  model_registry: {absolute_registry}\n"
            f"  reports: {absolute_reports}\n"
        )

        context = load_project_context(configuration)

        self.assertEqual(context.images_directory, absolute_images)
        self.assertEqual(context.labels_directory, absolute_labels)
        self.assertEqual(context.model_registry, absolute_registry)
        self.assertEqual(context.reports_directory, absolute_reports)

    def test_missing_configuration_fails_with_path(self) -> None:
        missing = self.root / "missing.yaml"

        with self.assertRaisesRegex(ProjectContextError, "Cannot read project configuration"):
            load_project_context(missing)

    def test_missing_required_setting_fails_clearly(self) -> None:
        configuration = self.write_config(
            "project:\n"
            "  organization_id: org\n"
            "paths:\n"
            "  images: images\n"
            "  labels: labels\n"
            "  model_registry: registry.json\n"
            "  reports: reports\n"
        )

        with self.assertRaisesRegex(ProjectContextError, "project.project_id"):
            load_project_context(configuration)

    def test_malformed_configuration_fails_clearly(self) -> None:
        configuration = self.write_config(
            "project: not-a-mapping\n"
            "paths:\n"
            "  images: images\n"
        )

        with self.assertRaisesRegex(ProjectContextError, "must define project: as a mapping"):
            load_project_context(configuration)

    def test_invalid_identifier_fails_clearly(self) -> None:
        configuration = self.write_config(
            "project:\n"
            "  organization_id: ../outside\n"
            "  project_id: project\n"
            "paths:\n"
            "  images: images\n"
            "  labels: labels\n"
            "  model_registry: registry.json\n"
            "  reports: reports\n"
        )

        with self.assertRaisesRegex(ProjectContextError, "Invalid project identifier"):
            load_project_context(configuration)

    def test_project_config_option_is_available_after_resource_commands(self) -> None:
        command_arguments = {
            "check-data": ["check-data"],
            "register-model": ["register-model", "--id", "model", "--path", "model.pt"],
            "verify-model": ["verify-model", "--id", "model"],
            "infer": ["infer", "--id", "model", "--image", "image.png"],
            "assess": ["assess", "--id", "model", "--image", "image.png"],
            "demo-model-tampering": ["demo-model-tampering"],
        }
        for command, arguments in command_arguments.items():
            with self.subTest(command=command):
                parsed = build_parser().parse_args(
                    [*arguments, "--project-config", "selected-project.yaml"]
                )
                self.assertEqual(parsed.project_config, Path("selected-project.yaml"))


if __name__ == "__main__":
    unittest.main()
