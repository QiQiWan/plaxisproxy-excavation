"""License-free fake API test for strict per-phase PLAXIS updates."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "plaxisproxy_excavation" / "plaxishelper" / "phase_properties.py"
SPEC = importlib.util.spec_from_file_location("phase_properties_only", SOURCE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

class FakeProp:
    def __init__(self):
        self.values = {}
    def __getitem__(self, phase):
        return SimpleNamespace(value=self.values[phase])

class FakeInput:
    def set(self, prop, phase, value):
        prop.values[phase] = value

class PhaseTests(unittest.TestCase):
    def test_unique_phase_update_and_readback(self):
        phase_a, phase_b = object(), object()
        feature = SimpleNamespace(PrestressForce=FakeProp())
        api = FakeInput()
        MOD.set_phase_property(api, feature, phase_a, "PrestressForce", 800)
        MOD.set_phase_property(api, feature, phase_b, "PrestressForce", 900)
        self.assertEqual(feature.PrestressForce.values[phase_a], 800)
        self.assertEqual(feature.PrestressForce.values[phase_b], 900)

    def test_unmapped_property_fails_without_fallback(self):
        with self.assertRaises(MOD.PhasePropertyError):
            MOD.set_phase_property(FakeInput(), object(), object(), "Missing", 1)

    def test_sign_must_be_verified(self):
        obj = SimpleNamespace(AdjustPrestress=FakeProp(), PrestressForce=FakeProp())
        with self.assertRaises(MOD.PhasePropertyError):
            MOD.set_node_to_node_prestress(FakeInput(), obj, object(), 100)

    def test_applied_prestress_is_not_actual_axial_force(self):
        obj = SimpleNamespace(AdjustPrestress=FakeProp(), PrestressForce=FakeProp())
        result = MOD.set_node_to_node_prestress(FakeInput(), obj, object(), -120,
                                                compression_sign_verified=True)
        self.assertIsNone(result["actual_axial_force_kn"])
        self.assertTrue(result["requires_output_check"])

    def test_bad_native_return_is_detected(self):
        obj = SimpleNamespace(PrestressForce=FakeProp())
        class BadInput:
            def set(self, prop, phase, value):
                prop.values[phase] = -9999
        with self.assertRaises(MOD.PhasePropertyError):
            MOD.set_phase_property(BadInput(), obj, object(), "PrestressForce", 20)

if __name__ == "__main__":
    unittest.main()
