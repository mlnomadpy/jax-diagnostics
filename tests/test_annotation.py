from pathlib import Path
import subprocess
import sys
import unittest

from jax_diagnostics import annotate_errors


class AnnotationTest(unittest.TestCase):
    def test_preserves_exception_instance_and_cause(self):
        cause = ValueError("original cause")
        error = RuntimeError("RESOURCE_EXHAUSTED")
        error.add_note("An existing backend note")
        with self.assertRaises(RuntimeError) as caught:
            with annotate_errors(guide_base="https://example.org/mirror/"):
                raise error from cause
        self.assertIs(caught.exception, error)
        self.assertIs(error.__cause__, cause)
        self.assertEqual(str(error), "RESOURCE_EXHAUSTED")
        self.assertIn("https://example.org/mirror/references/runtime.md#jax-mem-001", error.__notes__[-1])
        self.assertEqual(error.__notes__[0], "An existing backend note")

    def test_nested_wrappers_add_one_note(self):
        with self.assertRaises(RuntimeError) as caught:
            with annotate_errors(), annotate_errors():
                raise RuntimeError("out of memory")
        self.assertEqual(len(caught.exception.__notes__), 1)

    def test_unrecognized_exception_is_untouched(self):
        with self.assertRaises(ValueError) as caught:
            with annotate_errors():
                raise ValueError("unrecognized")
        self.assertFalse(hasattr(caught.exception, "__notes__"))

    def test_failure_while_annotating_preserves_original_exception(self):
        class BadMessage(ValueError):
            def add_note(self, note):
                raise RuntimeError("annotation failure")

        error = BadMessage("out of memory")
        with self.assertRaises(BadMessage) as caught:
            with annotate_errors():
                raise error
        self.assertIs(caught.exception, error)

    def test_normal_entrypoint_prints_note_and_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-c", """
from jax_diagnostics import annotate_errors
with annotate_errors():
    raise RuntimeError("RESOURCE_EXHAUSTED")
"""],
            cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("RuntimeError: RESOURCE_EXHAUSTED", result.stderr)
        self.assertIn("[JAX-MEM-001]", result.stderr)
