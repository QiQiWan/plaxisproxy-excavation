"""License-free contract tests; these do not validate a PLAXIS solver."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src/plaxisproxy_excavation/plaxishelper/phase_properties.py'
SPEC = importlib.util.spec_from_file_location('phase_properties_only', SOURCE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FakeProp:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.read_error = False

    def __getitem__(self, phase):
        if self.read_error:
            raise RuntimeError('unreadable phase')
        return SimpleNamespace(value=self.values[phase])


class FakeInput:
    def __init__(self, hook=None):
        self.calls = []
        self.hook = hook

    def set(self, prop, phase, value):
        self.calls.append((prop, phase, value))
        if self.hook:
            self.hook(self, prop, phase, value)
        prop.values[phase] = value


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = object(), object()
        self.obj = SimpleNamespace(
            AdjustPrestress=FakeProp({self.a: False, self.b: True}),
            PrestressForce=FakeProp({self.a: 0.0, self.b: -50.0}))
        self.api = FakeInput()

    def apply(self, **kwargs):
        return MOD.set_node_to_node_prestress(
            self.api, self.obj, self.a, -120,
            compression_sign_verified=True, **kwargs)

    def assert_original(self):
        self.assertIs(self.obj.AdjustPrestress.values[self.a], False)
        self.assertEqual(self.obj.PrestressForce.values[self.a], 0)

    def test_unique_phase_update_and_readback(self):
        MOD.set_phase_property(self.api, self.obj, self.a, 'PrestressForce', 800)
        MOD.set_phase_property(self.api, self.obj, self.b, 'PrestressForce', 900)
        self.assertEqual(self.obj.PrestressForce.values, {self.a: 800, self.b: 900})

    def test_unmapped_property_fails_without_fallback(self):
        with self.assertRaises(MOD.PhasePropertyError):
            MOD.set_phase_property(self.api, self.obj, self.a, 'Missing', 1)
        self.assertFalse(self.api.calls)

    def test_sign_must_be_explicit_true(self):
        for flag in (False, None, 1, 'yes'):
            with self.subTest(flag=flag), self.assertRaises(MOD.PhasePropertyError):
                MOD.set_node_to_node_prestress(self.api, self.obj, self.a, -120,
                                                compression_sign_verified=flag)
        self.assertFalse(self.api.calls)

    def test_command_is_not_actual_force(self):
        result = self.apply()
        self.assertEqual(result['command_prestress_kn'], -120)
        self.assertIsNone(result['actual_axial_force_kn'])
        self.assertTrue(result['requires_output_check'])

    def test_second_write_failure_restores_pair(self):
        def fail(api, prop, phase, value):
            if len(api.calls) == 2:
                raise RuntimeError('force write failed')
        self.api.hook = fail
        with self.assertRaises(MOD.PhasePropertyError) as ctx:
            self.apply()
        self.assert_original()
        self.assertTrue(ctx.exception.rollback_verified)
        self.assertFalse(ctx.exception.recovery_required)
        self.assertIsInstance(ctx.exception.__cause__, Exception)

    def test_setter_mutates_then_raises_is_restored(self):
        def fail(api, prop, phase, value):
            if len(api.calls) == 2:
                prop.values[phase] = value
                raise TimeoutError('response lost after apply')
        self.api.hook = fail
        with self.assertRaises(MOD.PhasePropertyError):
            self.apply()
        self.assert_original()

    def test_first_write_mutates_then_raises_is_restored(self):
        def fail(api, prop, phase, value):
            if len(api.calls) == 1:
                prop.values[phase] = value
                raise RuntimeError('lost response')
        self.api.hook = fail
        with self.assertRaises(MOD.PhasePropertyError):
            self.apply()
        self.assert_original()

    def test_restore_failure_is_reported_but_other_restore_continues(self):
        def fail(api, prop, phase, value):
            if len(api.calls) == 2:
                prop.values[phase] = value
                raise RuntimeError('failed after apply')
            if len(api.calls) == 3:
                raise RuntimeError('restore force failed')
        self.api.hook = fail
        with self.assertRaises(MOD.PhasePropertyError) as ctx:
            self.apply()
        self.assertFalse(self.obj.AdjustPrestress.values[self.a])
        self.assertEqual(self.obj.PrestressForce.values[self.a], -120)
        self.assertFalse(ctx.exception.rollback_verified)
        self.assertTrue(ctx.exception.recovery_required)
        self.assertTrue(ctx.exception.rollback_errors)

    def test_snapshot_failure_causes_zero_writes(self):
        self.obj.PrestressForce.read_error = True
        with self.assertRaises(MOD.PhasePropertyError):
            self.apply()
        self.assertFalse(self.api.calls)

    def test_invalid_snapshot_scalar_causes_zero_writes(self):
        for value in (float('nan'), float('inf'), True, object()):
            self.obj.PrestressForce.values[self.a] = value
            with self.subTest(value=value), self.assertRaises(MOD.PhasePropertyError):
                self.apply()
        self.assertFalse(self.api.calls)

    def test_unknown_boolean_snapshot_is_rejected(self):
        for value in ('unknown', 2, -1, None):
            self.obj.AdjustPrestress.values[self.a] = value
            with self.subTest(value=value), self.assertRaises(MOD.PhasePropertyError):
                self.apply()
        self.assertFalse(self.api.calls)

    def test_none_phase_is_rejected_before_write(self):
        with self.assertRaises(MOD.PhasePropertyError):
            MOD.set_phase_property(self.api, self.obj, None, 'PrestressForce', 10)
        self.assertFalse(self.api.calls)

    def test_proxy_attribute_exception_is_wrapped(self):
        class Broken:
            @property
            def PrestressForce(self):
                raise RuntimeError('stale proxy')
        with self.assertRaises(MOD.PhasePropertyError):
            MOD.set_phase_property(self.api, Broken(), self.a, 'PrestressForce', 10)
        self.assertFalse(self.api.calls)

    def test_proxy_value_exception_is_wrapped(self):
        class Broken:
            @property
            def value(self):
                raise RuntimeError('disconnected')
        class Prop(FakeProp):
            def __getitem__(self, phase):
                return Broken()
        self.obj.PrestressForce = Prop()
        with self.assertRaises(MOD.PhasePropertyError):
            self.apply()
        self.assertFalse(self.api.calls)

    def test_missing_second_property_causes_zero_writes(self):
        del self.obj.PrestressForce
        with self.assertRaises(MOD.PhasePropertyError):
            self.apply()
        self.assertFalse(self.api.calls)

    def test_repeated_write_is_noop(self):
        self.apply()
        self.api.calls.clear()
        self.apply()
        self.assertEqual(self.api.calls, [])

    def test_explicit_guard_phase_unchanged(self):
        self.apply(guard_phases=(self.b,))
        self.assertIs(self.obj.AdjustPrestress.values[self.b], True)
        self.assertEqual(self.obj.PrestressForce.values[self.b], -50)
        self.assertTrue(all(call[1] is self.a for call in self.api.calls))

    def test_cross_phase_pollution_is_detected_without_guard_writes(self):
        def pollute(api, prop, phase, value):
            if len(api.calls) == 2:
                prop.values[self.b] = -999
        self.api.hook = pollute
        with self.assertRaises(MOD.PhasePropertyError) as ctx:
            self.apply(guard_phases=(self.b,))
        self.assert_original()
        self.assertTrue(ctx.exception.recovery_required)
        self.assertTrue(all(call[1] is self.a for call in self.api.calls))

    def test_final_pair_readback_detects_first_property_drift(self):
        def drift(api, prop, phase, value):
            if len(api.calls) == 2:
                self.obj.AdjustPrestress.values[phase] = False
        self.api.hook = drift
        with self.assertRaises(MOD.PhasePropertyError):
            self.apply()
        self.assert_original()

    def test_unknown_boolean_readback_does_not_match_false(self):
        def corrupt(api, prop, phase, value):
            if len(api.calls) == 1:
                prop.values[phase] = 'unknown'
                raise RuntimeError('unknown readback after partial apply')
        self.api.hook = corrupt
        with self.assertRaises(MOD.PhasePropertyError):
            MOD.set_phase_property(self.api, self.obj, self.a, 'AdjustPrestress', True)
        self.assert_original()

    def test_single_property_readback_mismatch_restores_original(self):
        def corrupt(api, prop, phase, value):
            if len(api.calls) == 1:
                prop.values[phase] = -999
                raise RuntimeError('bad apply')
        self.api.hook = corrupt
        with self.assertRaises(MOD.PhasePropertyError):
            MOD.set_phase_property(self.api, self.obj, self.a, 'PrestressForce', 20)
        self.assert_original()

    def test_invalid_forces_rejected_before_write(self):
        for value in (True, 0, float('inf'), float('nan'), '120'):
            with self.subTest(value=value), self.assertRaises(MOD.PhasePropertyError):
                MOD.set_node_to_node_prestress(self.api, self.obj, self.a, value,
                                               compression_sign_verified=True)
        self.assertFalse(self.api.calls)

    def test_invalid_tolerance_rejected_before_write(self):
        for value in (-1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(MOD.PhasePropertyError):
                MOD.set_phase_property(self.api, self.obj, self.a, 'PrestressForce', 10,
                                       abs_tol=value)
        self.assertFalse(self.api.calls)


    def test_multi_property_restore_accepts_false_and_zero(self):
        self.apply()
        MOD.set_phase_properties(self.api, self.obj, self.a,
                                 {'AdjustPrestress': False, 'PrestressForce': 0},
                                 guard_phases=(self.b,))
        self.assert_original()

    def test_actual_readback_mismatch_is_compensated(self):
        class BadRead(FakeProp):
            def __getitem__(prop, phase):
                value = prop.values[phase]
                return SimpleNamespace(value=-999 if value == -120 else value)
        self.obj.PrestressForce = BadRead({self.a: 0, self.b: -50})
        with self.assertRaises(MOD.PhasePropertyError) as ctx:
            self.apply()
        self.assert_original()
        self.assertTrue(ctx.exception.rollback_verified)

    def test_unreadable_final_recovery_is_uncertain(self):
        def break_read(api, prop, phase, value):
            if len(api.calls) == 2:
                prop.read_error = True
        self.api.hook = break_read
        with self.assertRaises(MOD.PhasePropertyError) as ctx:
            self.apply()
        self.assertTrue(ctx.exception.recovery_required)
        self.assertFalse(ctx.exception.rollback_verified)

    def test_false_readback_rejects_unknown_string(self):
        self.obj.AdjustPrestress.values[self.a] = True
        class BadRead(FakeProp):
            def __getitem__(prop, phase):
                value = prop.values[phase]
                return SimpleNamespace(value='unknown' if value is False else value)
        self.obj.AdjustPrestress = BadRead({self.a: True})
        with self.assertRaises(MOD.PhasePropertyError) as ctx:
            MOD.set_phase_property(self.api, self.obj, self.a, 'AdjustPrestress', False)
        self.assertIs(self.obj.AdjustPrestress.values[self.a], True)
        self.assertTrue(ctx.exception.rollback_verified)

    def test_unknown_phase_is_rejected_before_writes(self):
        with self.assertRaises(MOD.PhasePropertyError):
            MOD.set_node_to_node_prestress(self.api, self.obj, object(), -120,
                                          compression_sign_verified=True)
        self.assertFalse(self.api.calls)

    def test_guard_snapshot_failure_is_rejected_before_writes(self):
        with self.assertRaises(MOD.PhasePropertyError):
            self.apply(guard_phases=(object(),))
        self.assertFalse(self.api.calls)

    def test_first_setter_coupled_side_effect_restores_entire_snapshot(self):
        def coupled(api, prop, phase, value):
            if len(api.calls) == 1:
                prop.values[phase] = value
                self.obj.PrestressForce.values[phase] = -999
                raise RuntimeError('coupled apply lost response')
        self.api.hook = coupled
        with self.assertRaises(MOD.PhasePropertyError) as ctx:
            self.apply()
        self.assert_original()
        self.assertTrue(ctx.exception.rollback_verified)
        self.assertTrue(any(prop is self.obj.PrestressForce and value == 0
                            for prop, phase, value in self.api.calls))

    def test_interrupt_compensates_then_preserves_interruption(self):
        for kind in (KeyboardInterrupt, SystemExit):
            self.setUp()
            def interrupt(api, prop, phase, value):
                if len(api.calls) == 2:
                    prop.values[phase] = value
                    raise kind('cancelled after apply')
            self.api.hook = interrupt
            with self.subTest(kind=kind), self.assertRaises(kind) as ctx:
                self.apply()
            self.assert_original()
            self.assertTrue(ctx.exception.phase_property_error.rollback_verified)

    def test_interrupt_during_recovery_still_restores_other_property(self):
        def fail(api, prop, phase, value):
            if len(api.calls) == 2:
                prop.values[phase] = value
                raise RuntimeError('second write lost response')
            if len(api.calls) == 3:
                raise KeyboardInterrupt('interrupted recovery')
        self.api.hook = fail
        with self.assertRaises(KeyboardInterrupt) as ctx:
            self.apply()
        self.assertFalse(self.obj.AdjustPrestress.values[self.a])
        self.assertTrue(ctx.exception.phase_property_error.recovery_required)

    def test_lost_response_on_restore_can_still_be_verified(self):
        def fail(api, prop, phase, value):
            if len(api.calls) in (2, 3):
                prop.values[phase] = value
                raise TimeoutError('applied but lost response')
        self.api.hook = fail
        with self.assertRaises(MOD.PhasePropertyError) as ctx:
            self.apply()
        self.assert_original()
        self.assertTrue(ctx.exception.rollback_verified)
        self.assertTrue(ctx.exception.rollback_errors)


if __name__ == '__main__':
    unittest.main()
