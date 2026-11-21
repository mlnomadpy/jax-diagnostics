import json
from pathlib import Path
import subprocess
import sys
import unittest

from jax_diagnostics._routing import ERRORS, SYMPTOMS, diagnose

ROOT = Path(__file__).resolve().parent.parent


class RoutingTest(unittest.TestCase):
    def test_every_public_exception_has_a_route(self):
        for name, route in ERRORS.items():
            for error in (name, f"jax.errors.{name}: example"):
                with self.subTest(error=error):
                    result = diagnose(error)
                    self.assertEqual((result.id, result.match), (route.id, "exception"))

    def test_all_route_links_resolve(self):
        for route in [*ERRORS.values(), *(route for _, route in SYMPTOMS)]:
            link, anchor = route.link.split("#")
            headings = (ROOT / link).read_text().splitlines()
            self.assertIn(f"## {anchor.upper()}", headings)

    def test_traceback_prefers_exception_over_source_text(self):
        result = diagnose(
            'Traceback (most recent call last):\n'
            '  File "main.py", line 2, in run\n'
            '    log("OOM and nan")\n'
            'jax.errors.TracerArrayConversionError: array conversion failed'
        )
        self.assertEqual(result.id, "JAX-TRACE-005")

    def test_last_specific_cause_for_chained_traceback(self):
        result = diagnose(
            "jax.errors.JAXTypeError: dtype mismatch\n\n"
            "During handling of the above exception, another exception occurred:\n\n"
            "jax.errors.TracerBoolConversionError: traced bool"
        )
        self.assertEqual(result.id, "JAX-TRACE-003")

    def test_runtime_status_specializes_generic_error(self):
        result = diagnose("jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED")
        self.assertEqual((result.id, result.match), ("JAX-MEM-001", "symptom"))

    def test_symptoms_route_and_unknown_abstains(self):
        self.assertEqual(diagnose("unexpected recompiles").id, "JAX-COMPILE-001")
        self.assertEqual(diagnose("collective timed out").id, "JAX-DIST-001")
        self.assertIsNone(diagnose("ValueError: something else"))
        self.assertIsNone(diagnose(""))

    def test_static_slice_messages_route_to_dynamic_slice_guide(self):
        for message in (
            "IndexError: Slice entries must be static integers. Got slice(None, JitTracer(int32[]), None)",
            "IndexError: Array slice indices must have static start/stop/step to be used with NumPy indexing syntax.",
        ):
            with self.subTest(message=message):
                self.assertEqual(diagnose(message).id, "JAX-INDEX-002")

    def test_reverse_mode_dynamic_loop_has_a_specific_route(self):
        message = (
            "ValueError: Reverse-mode differentiation does not work for "
            "lax.while_loop or lax.fori_loop with dynamic start/stop values. "
            "Try using lax.scan, or using fori_loop with static start/stop."
        )
        self.assertEqual(diagnose(message).id, "JAX-GRAD-002")

    def test_unrelated_index_and_loop_errors_do_not_match(self):
        self.assertIsNone(diagnose("IndexError: list index out of range"))
        self.assertIsNone(diagnose("ValueError: lax.while_loop body returned an incompatible value"))

    def test_cli_json_reads_standard_input(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/diagnose.py"), "--json"],
            input="jax.errors.KeyReuseError: key already consumed",
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(json.loads(result.stdout)["id"], "JAX-RNG-001")

    def test_cli_unknown_exits_with_status_one(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/diagnose.py"), "--json", "unrecognized"],
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout), {"match": None})
