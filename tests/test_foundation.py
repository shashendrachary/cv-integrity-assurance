"""Small standard-library tests for the project foundation."""

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "assurance.py"


class FoundationCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLI_PATH), *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )

    def test_help_displays_usage_without_external_services(self) -> None:
        result = self.run_cli("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        self.assertIn("CV-Integrity-Assurance", result.stdout)

    def test_default_command_displays_foundation_status(self) -> None:
        result = self.run_cli()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Project: CV-Integrity-Assurance", result.stdout)
        self.assertIn("Initialization status: ready", result.stdout)
        self.assertIn("Planned capabilities (not implemented yet):", result.stdout)
        self.assertIn("Dataset integrity checks", result.stdout)


if __name__ == "__main__":
    unittest.main()
