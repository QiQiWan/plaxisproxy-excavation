"""Contracts for the combined, installed APIs; never connect to a real solver."""
import importlib
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class IntegratedApiTests(unittest.TestCase):
    def test_installed_public_imports_do_not_load_vendor_client(self):
        code = '''
import sys
class NoVendor:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'plxscripting' or fullname.startswith('plxscripting.'):
            raise AssertionError('Core import tried loading the optional vendor client')
sys.meta_path.insert(0, NoVendor())
import plaxisproxy_excavation
from plaxisproxy_excavation.builder.excavation_builder import ExcavationBuilder
from plaxisproxy_excavation.plaxishelper import (
    PlaxisRunner, PlaxisOutput, PhasePropertyError, set_phase_properties,
    set_phase_property, set_node_to_node_prestress,
    PhaseResultError, read_node_to_node_axial_force,
)
assert not any(k.startswith('third_party') for k in sys.modules)
'''
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, '-c', code], cwd=cwd,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_vendor_has_actionable_connect_error(self):
        scripting = importlib.import_module('plaxisproxy_excavation.plaxishelper.scripting')
        missing = ModuleNotFoundError("No module named 'plxscripting'", name='plxscripting')
        with patch.object(scripting, 'import_module', side_effect=missing):
            with self.assertRaisesRegex(ImportError, 'matching.*PLAXIS'):
                scripting.new_server('localhost', 10000, password='test-only')

    def test_broken_vendor_dependency_error_is_not_hidden(self):
        scripting = importlib.import_module('plaxisproxy_excavation.plaxishelper.scripting')
        missing = ModuleNotFoundError('Missing client dependency', name='vendor_dependency')
        with patch.object(scripting, 'import_module', side_effect=missing):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                scripting.new_server('localhost', 10000, password='test-only')
        self.assertIs(ctx.exception, missing)

    def test_connect_uses_vendor_easy_without_bundled_fallback(self):
        scripting = importlib.import_module('plaxisproxy_excavation.plaxishelper.scripting')
        seen = []
        easy = SimpleNamespace(new_server=lambda *a, **kw: seen.append((a, kw)) or ('s', 'g'))
        with patch.object(scripting, 'import_module', return_value=easy) as loader:
            self.assertEqual(scripting.new_server('localhost', 10000, password='test-only'), ('s', 'g'))
        loader.assert_called_once_with('plxscripting.easy')
        self.assertEqual(seen, [(('localhost', 10000), {'password': 'test-only'})])

    def test_empty_or_disabled_phase_plan_does_not_start_calculation(self):
        from plaxisproxy_excavation.builder.excavation_builder import ExcavationBuilder
        for phases in ([], [SimpleNamespace(plx_id=SimpleNamespace(ShouldCalculate=False))]):
            calls = []
            builder = ExcavationBuilder.__new__(ExcavationBuilder)
            builder.excavation_object = SimpleNamespace(phases=phases)
            builder.App = SimpleNamespace(calculate=lambda: calls.append('calculate'))
            builder._calc_done = True
            with self.subTest(phases=phases), self.assertRaisesRegex(RuntimeError, 'No phases'):
                builder.calculate()
            self.assertEqual(calls, [])
            self.assertFalse(builder._calc_done)

    def test_public_runner_does_not_hide_broken_client_initialization(self):
        from plaxisproxy_excavation.plaxishelper import PlaxisRunner
        scripting = importlib.import_module('plaxisproxy_excavation.plaxishelper.scripting')
        for error in (RuntimeError('client initialization failed'), OSError('client ABI mismatch')):
            with patch.object(scripting, 'import_module', side_effect=error):
                with self.assertRaises(type(error)) as ctx:
                    PlaxisRunner('test-only', 'localhost', 10000).connect()
                self.assertIs(ctx.exception, error)
