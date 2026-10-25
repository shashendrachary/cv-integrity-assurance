"""Offline safety tests for repeatable integrity demonstrations."""

import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from src.demos import run_data_integrity_demo, run_model_tampering_demo
from src.model.integrity import register_model


class SecurityDemoTests(unittest.TestCase):
    def test_model_demo_only_modifies_temporary_copy_and_blocks_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_model = root / "registered-model.bin"
            original_model.write_bytes(b"safe source model bytes")
            registry = root / "registry.json"
            register_model("demo-source", original_model, registry)
            original_digest = hashlib.sha256(original_model.read_bytes()).hexdigest()
            before_paths = set(root.iterdir())
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                expected, actual, status, blocked = run_model_tampering_demo(
                    "demo-source", registry
                )

            self.assertEqual(expected, original_digest)
            self.assertNotEqual(actual, expected)
            self.assertEqual(status, "SUSPICIOUS")
            self.assertTrue(blocked)
            self.assertIn("BLOCKED before loader", output.getvalue())
            self.assertEqual(
                hashlib.sha256(original_model.read_bytes()).hexdigest(),
                original_digest,
            )
            self.assertEqual(set(root.iterdir()), before_paths)

    def test_data_demo_reports_corrupt_duplicate_label_and_pair_findings(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            report = run_data_integrity_demo()

        combined_output = output.getvalue() + report
        self.assertIn("corrupt or unreadable image", combined_output)
        self.assertIn("Byte-identical duplicate images", combined_output)
        self.assertIn("coordinates and dimensions must be numeric", combined_output)
        self.assertIn("Image has no corresponding label file", combined_output)
        self.assertIn("Temporary data-integrity demonstration", combined_output)


if __name__ == "__main__":
    unittest.main()
