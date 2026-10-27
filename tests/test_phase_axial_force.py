"""Explicit Output contract; no solver/license is used."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src/plaxisproxy_excavation/plaxishelper/phase_results.py'
# Missing implementation is an expected RED for this new API.


class OutputTests(unittest.TestCase):
    def reader(self):
        spec = importlib.util.spec_from_file_location('phase_results_only', SOURCE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_explicit_support_phase_result_type_and_values(self):
        mod = self.reader()
        support, phase, result_type = object(), object(), object()
        calls = []
        def getresults(*args):
            calls.append(args)
            return [-120.0, -118.0]
        result = mod.read_node_to_node_axial_force(
            SimpleNamespace(getresults=getresults), support, phase,
            result_type=result_type, force_unit='kN')
        self.assertEqual(calls, [(support, phase, result_type, 'node')])
        self.assertEqual(result['axial_force_kn'], [-120.0, -118.0])
        self.assertEqual(result['source'], 'PLAXIS Output')
        self.assertNotIn('command_prestress_kn', result)

    def test_reject_empty_text_scalar_nonfinite_and_bool_results(self):
        mod = self.reader()
        for raw in ([], '', 'not calculated', -120, [float('nan')],
                    [float('inf')], [True], ['120'], [object()], {-120: 'server error'}):
            api = SimpleNamespace(getresults=lambda *args: raw)
            with self.subTest(raw=raw), self.assertRaises(mod.PhaseResultError):
                mod.read_node_to_node_axial_force(api, object(), object(),
                                                 result_type=object(), force_unit='kN')

    def test_missing_handles_and_unconfirmed_unit_do_not_query(self):
        mod = self.reader()
        calls = []
        api = SimpleNamespace(getresults=lambda *args: calls.append(args))
        for support, phase, kind, unit in ((None, object(), object(), 'kN'),
                                           (object(), None, object(), 'kN'),
                                           (object(), object(), None, 'kN'),
                                           (object(), object(), object(), 'N')):
            with self.assertRaises(mod.PhaseResultError):
                mod.read_node_to_node_axial_force(api, support, phase,
                                                 result_type=kind, force_unit=unit)
        self.assertEqual(calls, [])

    def test_query_error_is_wrapped_without_alternate_phase_fallback(self):
        mod = self.reader()
        calls = []
        def fail(*args):
            calls.append(args)
            raise RuntimeError('phase not calculated')
        with self.assertRaises(mod.PhaseResultError):
            mod.read_node_to_node_axial_force(SimpleNamespace(getresults=fail),
                                             object(), object(), result_type=object(),
                                             force_unit='kN')
        self.assertEqual(len(calls), 1)
