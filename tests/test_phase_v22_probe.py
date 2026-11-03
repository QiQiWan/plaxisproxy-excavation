"""Probe orchestration contracts using a fake Input, never Windows/V22 execution."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    'probe_fake_helpers', Path(__file__).with_name('test_phase_property_gateway.py'))
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
FakeInput, FakeProp = HELPERS.FakeInput, HELPERS.FakeProp
SOURCE = Path(__file__).resolve().parents[1] / 'scripts/verify_v22_phase_prestress.py'


class ProbeTests(unittest.TestCase):
    def run_probe(self, failure='none'):
        spec = importlib.util.spec_from_file_location('v22_probe', SOURCE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        a, b = object(), object()
        obj = SimpleNamespace(AdjustPrestress=FakeProp({a: False, b: False}),
                              PrestressForce=FakeProp({a: 0, b: 0}))
        report = {}
        checkpoints = []
        mod.probe_input(FakeInput(), obj, a, (b,), -120, failure, report,
                        lambda: checkpoints.append(dict(report)))
        self.assertEqual(obj.AdjustPrestress.values, {a: False, b: False})
        self.assertEqual(obj.PrestressForce.values, {a: 0, b: 0})
        self.assertTrue(report['cleanup_verified'])
        self.assertTrue(report['input_contract_passed'])
        self.assertTrue(checkpoints)
        return report

    def test_success_probe_restores_original_and_checks_idempotence(self):
        report = self.run_probe()
        self.assertTrue(report['repeat_was_noop'])

    def test_failure_before_apply_is_expected_and_recovers(self):
        report = self.run_probe('before')
        self.assertTrue(report['expected_failure']['rollback_verified'])

    def test_failure_after_apply_is_expected_and_recovers(self):
        report = self.run_probe('after')
        self.assertTrue(report['expected_failure']['rollback_verified'])

    def test_first_write_server_error_must_not_pass_injection_probe(self):
        spec = importlib.util.spec_from_file_location('v22_probe_first_error', SOURCE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        a, b = object(), object()
        obj = SimpleNamespace(AdjustPrestress=FakeProp({a: False, b: False}),
                              PrestressForce=FakeProp({a: 0, b: 0}))
        def fail_first(api, prop, phase, value):
            if len(api.calls) == 1:
                raise RuntimeError('real first-set server error')
        report = {}
        with self.assertRaises(mod.PROPS.PhasePropertyError):
            mod.probe_input(FakeInput(fail_first), obj, a, (b,), -120, 'before',
                            report, lambda: None)
        self.assertFalse(report['input_contract_passed'])
        self.assertNotIn('expected_failure', report)
