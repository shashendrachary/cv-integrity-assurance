"""Temporary-file tests for image and YOLO-label integrity checks."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.integrity.data_checks import (
    format_data_check,
    scan_dataset,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "assurance.py"


class Sha256Tests(unittest.TestCase):
    def test_sha256_matches_known_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "known.txt"
            file_path.write_bytes(b"integrity assurance")

            self.assertEqual(
                sha256_file(file_path),
                hashlib.sha256(b"integrity assurance").hexdigest(),
            )

    def test_missing_file_raises_clear_os_error(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "assurance-missing-file"

        with self.assertRaisesRegex(OSError, "Unable to read file for SHA-256"):
            sha256_file(missing_path)

    def test_directory_is_reported_as_unreadable_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(OSError, "Unable to read file for SHA-256"):
                sha256_file(Path(temporary_directory))


class DatasetCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.images.mkdir()
        self.labels.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_image(self, relative_path: str, color: tuple[int, int, int]) -> Path:
        image_path = self.images / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color).save(image_path, format="PNG")
        return image_path

    def test_corrupt_image_is_invalid(self) -> None:
        (self.images / "broken.png").write_bytes(b"not an image")

        result = scan_dataset(self.images, self.labels)

        self.assertEqual(result.images_examined, 1)
        self.assertEqual(result.valid_images, 0)
        self.assertEqual(len(result.invalid_images), 1)
        self.assertIn("corrupt or unreadable image", result.invalid_images[0])

    def test_empty_image_is_invalid(self) -> None:
        (self.images / "empty.jpg").write_bytes(b"")

        result = scan_dataset(self.images, self.labels)

        self.assertEqual(len(result.invalid_images), 1)
        self.assertIn("file is empty", result.invalid_images[0])

    def test_duplicate_images_are_grouped_by_identical_bytes(self) -> None:
        first = self.write_image("first.png", (255, 0, 0))
        second = self.images / "copy.png"
        second.write_bytes(first.read_bytes())

        result = scan_dataset(self.images, self.labels)

        self.assertEqual(result.valid_images, 2)
        self.assertEqual(len(result.duplicate_groups), 1)
        self.assertEqual(set(result.duplicate_groups[0]), {str(first), str(second)})

    def test_unsupported_image_extension_is_reported(self) -> None:
        (self.images / "notes.bin").write_bytes(b"some contents")

        result = scan_dataset(self.images, self.labels)

        self.assertEqual(result.images_examined, 1)
        self.assertEqual(result.valid_images, 0)
        self.assertIn("unsupported image file extension", result.invalid_images[0])

    def test_label_boundaries_empty_file_and_malformed_lines(self) -> None:
        self.write_image("boundary.png", (0, 255, 0))
        (self.labels / "boundary.txt").write_text(
            "0 0 1 1 0.25\n3 1 0 0.01 1\n", encoding="utf-8"
        )
        (self.labels / "empty.txt").write_text("", encoding="utf-8")
        (self.labels / "bad.txt").write_text(
            "0 -0.1 0.5 0.2 0.2\n"
            "1 0.5 0.5 0 0.2\n"
            "-1 0.5 0.5 0.2 0.2\n"
            "2 0.5 nan 0.2 0.2\n"
            "3 0.5 0.5 0.2\n",
            encoding="utf-8",
        )

        result = scan_dataset(self.images, self.labels)

        self.assertEqual(result.valid_label_files, 2)
        self.assertEqual(len(result.invalid_label_files), 1)
        self.assertEqual(len(result.empty_label_files), 1)
        self.assertEqual(len(result.errors), 5)
        self.assertTrue(any(":5:" in error for error in result.errors))
        self.assertTrue(
            any("valid and represents zero objects" in item for item in result.warnings)
        )

    def test_non_numeric_class_and_infinite_values_are_invalid(self) -> None:
        (self.labels / "invalid.txt").write_text(
            "1.0 0.5 0.5 0.2 0.2\n"
            "0 inf 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

        result = scan_dataset(self.images, self.labels)

        self.assertEqual(len(result.invalid_label_files), 1)
        self.assertEqual(len(result.errors), 2)
        self.assertIn("class_id must be a non-negative integer", result.errors[0])
        self.assertIn("must be finite", result.errors[1])

    def test_missing_image_label_pairs_are_reported(self) -> None:
        self.write_image("paired.png", (10, 20, 30))
        (self.labels / "orphan.txt").write_text("0 0.5 0.5 1 1\n", encoding="utf-8")

        result = scan_dataset(self.images, self.labels)

        self.assertEqual(result.images_missing_labels, [str(self.images / "paired.png")])
        self.assertEqual(result.labels_missing_images, [str(self.labels / "orphan.txt")])

    def test_empty_or_absent_directories_report_no_data(self) -> None:
        empty_result = scan_dataset(self.images, self.labels)
        absent_result = scan_dataset(self.root / "missing-images", self.root / "missing-labels")

        self.assertFalse(empty_result.data_available)
        self.assertFalse(absent_result.data_available)
        self.assertIn("no data available", format_data_check(absent_result))
        self.assertEqual(empty_result.images_examined, 0)
        self.assertEqual(empty_result.valid_label_files, 0)


class DataCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLI_PATH), *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )

    def test_check_data_reports_empty_configured_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images_directory = root / "images"
            labels_directory = root / "labels"
            images_directory.mkdir()
            labels_directory.mkdir()

            result = self.run_cli(
                "check-data",
                "--images-dir",
                str(images_directory),
                "--labels-dir",
                str(labels_directory),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Data availability: no data available", result.stdout)
        self.assertIn("Images examined: 0", result.stdout)

    def test_check_data_operational_failure_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path_that_is_not_a_directory = Path(temporary_directory) / "file"
            path_that_is_not_a_directory.write_text("not a directory", encoding="utf-8")

            result = self.run_cli(
                "check-data",
                "--images-dir",
                str(path_that_is_not_a_directory),
                "--labels-dir",
                str(Path(temporary_directory) / "missing-labels"),
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("not a directory", result.stdout)


if __name__ == "__main__":
    unittest.main()
