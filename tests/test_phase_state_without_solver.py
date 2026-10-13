"""Verify calculation-state logic without a licensed PLAXIS runtime.

Methods are loaded from repository source AST; the live client is never imported.
This is a unit-level logic check, not a substitute for live API verification.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "plaxisproxy_excavation" / "plaxishelper" / "plaxisrunner.py"
BUILDER = ROOT / "src" / "plaxisproxy_excavation" / "builder" / "excavation_builder.py"


def read_method(path, class_name, name):
    module = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == class_name)
    method = next(x for x in owner.body if isinstance(x, ast.FunctionDef) and x.name == name)
    method.decorator_list = []
    isolated = ast.Module(body=[method], type_ignores=[])
    namespace = {"Phase": object}
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    return namespace[name]


class PhaseStateTests(unittest.TestCase):
    def setUp(self):
        self.is_calculated = read_method(RUNNER, "PlaxisRunner", "is_calculated")
        self.is_required = read_method(BUILDER, "ExcavationBuilder", "_phase_is_required")
        self.calculate = read_method(BUILDER, "ExcavationBuilder", "calculate")
        self.calculate_runner = read_method(RUNNER, "PlaxisRunner", "calculate")

    def phase(self, state, selected=True):
        return SimpleNamespace(name="Test", plx_id=SimpleNamespace(
            CalculationResult=SimpleNamespace(value=state), ShouldCalculate=selected))

    def builder(self, phases):
        app = SimpleNamespace(calculate=lambda: "OK",
                              is_calculated=lambda p: self.is_calculated(None, p))
        return SimpleNamespace(App=app, _calc_done=False,
                               list_project_phases=lambda: phases,
                               _phase_is_required=self.is_required)

    def test_only_converged_status_is_success(self):
        for state in (0, 2, None, "failure"):
            self.assertFalse(self.is_calculated(None, self.phase(state)))
        self.assertTrue(self.is_calculated(None, self.phase(1)))

    def test_requires_all_selected_phases(self):
        b = self.builder([self.phase(1), self.phase(2)])
        with self.assertRaisesRegex(RuntimeError, "not converged"):
            self.calculate(b)
        self.assertFalse(b._calc_done)

    def test_skips_explicitly_disabled_phase(self):
        b = self.builder([self.phase(1), self.phase(2, False)])
        self.calculate(b)
        self.assertTrue(b._calc_done)

    def test_strict_when_all_disabled(self):
        b = self.builder([self.phase(1, False)])
        with self.assertRaisesRegex(RuntimeError, "No phases are selected"):
            self.calculate(b)

    def test_remote_proxy_boolean_false_is_respected(self):
        b = self.builder([self.phase(1), self.phase(2, SimpleNamespace(value="False"))])
        self.calculate(b)
        self.assertTrue(b._calc_done)

    def test_calculation_command_feedback(self):
        ok = SimpleNamespace(g_i=SimpleNamespace(calculate=lambda: "OK"),
                             goto_stages=lambda: None)
        self.assertEqual(self.calculate_runner(ok), "OK")
        bad = SimpleNamespace(g_i=SimpleNamespace(calculate=lambda: "failed"),
                              goto_stages=lambda: None)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.calculate_runner(bad)


if __name__ == "__main__":
    unittest.main()
