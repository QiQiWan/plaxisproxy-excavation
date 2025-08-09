import unittest
from plaxisproxy_excavation.structures.load import (
    LoadStage, DistributionType, SignalType, LoadMultiplier,
    PointLoad, LineLoad, SurfaceLoad,
    DynPointLoad, DynLineLoad
)
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D
from enum import Enum

class TestLoadMultiplier(unittest.TestCase):
    def test_harmonic_signal_validation(self):
        """Harmonic signal requires valid amplitude (>=0), numeric phase, and frequency (>=0)."""
        # Negative amplitude should raise ValueError
        with self.assertRaises(ValueError):
            LoadMultiplier("L1", "c", SignalType.HARMONIC, amplitude=-5.0)
        # Non-numeric phase should raise ValueError
        with self.assertRaises(ValueError):
            LoadMultiplier("L2", "c", SignalType.HARMONIC, amplitude=1.0, phase="bad")
        # Negative frequency should raise ValueError
        with self.assertRaises(ValueError):
            LoadMultiplier("L3", "c", SignalType.HARMONIC, amplitude=1.0, phase=0.0, frequency=-1.0)
        # Valid harmonic inputs should set properties correctly
        lm = LoadMultiplier("HarmLM", "c", SignalType.HARMONIC, amplitude=2.5, phase=30.0, frequency=1.5)
        self.assertEqual(lm.signal_type, SignalType.HARMONIC)
        self.assertEqual(lm.amplitude, 2.5)
        self.assertEqual(lm.phase, 30.0)
        self.assertEqual(lm.frequency, 1.5)
        self.assertIsNone(lm.table_data)
        # __repr__ for harmonic signal includes amplitude, phase (deg symbol), and frequency
        repr_str = repr(lm)
        self.assertIn("LoadMultiplier(name='HarmLM'", repr_str)
        self.assertIn("signal='Harmonic'", repr_str)
        self.assertIn("amplitude=2.500", repr_str)
        self.assertIn("frequency=1.500 Hz", repr_str)

    def test_table_signal_validation(self):
        """Table signal requires table_data list of (time, value) pairs in ascending time order."""
        # Missing table_data should raise ValueError
        with self.assertRaises(ValueError):
            LoadMultiplier("T1", "c", SignalType.TABLE, table_data=None)
        # Non-list table_data should raise ValueError
        with self.assertRaises(ValueError):
            LoadMultiplier("T2", "c", SignalType.TABLE, table_data="not_a_list")
        # Entry that is not a numeric pair should raise ValueError
        with self.assertRaises(ValueError):
            LoadMultiplier("T3", "c", SignalType.TABLE, table_data=[(0.0, 1.0), (1.0, "a")])
        # Time not in ascending order should raise ValueError
        with self.assertRaises(ValueError):
            LoadMultiplier("T4", "c", SignalType.TABLE, table_data=[(1.0, 1.0), (0.5, 2.0)])
        # Valid table data should be accepted
        data = [(0.0, 0.0), (1.0, 5.0), (2.0, 5.0)]
        lm = LoadMultiplier("TableLM", "c", SignalType.TABLE, table_data=data)
        self.assertEqual(lm.signal_type, SignalType.TABLE)
        self.assertEqual(lm.table_data, data)
        self.assertIsNone(lm.amplitude)  # amplitude not used for TABLE
        # __repr__ for short table (<=4 points) includes full data list
        repr_str = repr(lm)
        self.assertIn("signal='Table'", repr_str)
        self.assertIn(str(data), repr_str)
        # Long table data should be truncated in __repr__
        long_data = [(i, float(i)) for i in range(6)]  # 6 points
        lm_long = LoadMultiplier("TableLM2", "c", SignalType.TABLE, table_data=long_data)
        repr_long = repr(lm_long)
        self.assertIn("...", repr_long)
        # Unsupported SignalType should raise NotImplementedError
        class FakeSignal(Enum):
            FAKE = "Fake"
        with self.assertRaises(NotImplementedError):
            LoadMultiplier("X", "c", FakeSignal.FAKE)

class TestPointLoad(unittest.TestCase):
    def test_force_and_moment_values(self):
        """PointLoad should store force and moment components correctly."""
        pt = Point(1, 2, 3)
        pl = PointLoad("P1", "c", point=pt, Fx=5.0, Fy=0.0, Fz=-1.0, Mx=2.0, My=0.0, Mz=0.0)
        # Default stage is STATIC and distribution is UNIFORM
        self.assertEqual(pl.stage, LoadStage.STATIC)
        self.assertEqual(pl.distribution, DistributionType.UNIFORM)
        # Point reference should be stored
        self.assertIs(pl.point, pt)
        # Forces and moments accessible via properties
        self.assertAlmostEqual(pl.Fx, 5.0)
        self.assertAlmostEqual(pl.Fy, 0.0)
        self.assertAlmostEqual(pl.Fz, -1.0)
        self.assertAlmostEqual(pl.Mx, 2.0)
        self.assertAlmostEqual(pl.My, 0.0)
        self.assertAlmostEqual(pl.Mz, 0.0)

    def test_repr_format(self):
        """__repr__ for PointLoad should include name, forces, and point location."""
        pt = Point(1, 1, 1)
        pl = PointLoad("LoadX", "c", point=pt, Fx=1.234, Fy=0.0, Fz=0.0)
        rep = repr(pl)
        self.assertIn("PointLoad", rep)
        self.assertIn("LoadX", rep)
        self.assertIn("Static", rep)
        self.assertIn("(1.0, 1.0, 1.0)", rep)  # point coordinates

class TestLineLoad(unittest.TestCase):
    def setUp(self):
        # A valid line (2 points) for testing
        self.valid_line = Line3D(PointSet([Point(0, 0, 0), Point(0, 1, 0)]))

    def test_invalid_parameters(self):
        """LineLoad should validate line length and distribution type."""
        short_line = Line3D(PointSet([Point(0, 0, 0)]))
        with self.assertRaises(ValueError):
            LineLoad("L1", "c", line=short_line)  # fewer than 2 points
        # Unsupported distribution type
        with self.assertRaises(ValueError):
            LineLoad("L2", "c", line=self.valid_line, distribution=DistributionType.PERPENDICULAR)

    def test_uniform_load_values(self):
        """Uniform LineLoad should store qx, qy, qz and default end values."""
        ll = LineLoad("L3", "c", line=self.valid_line, distribution=DistributionType.UNIFORM,
                      qx=10.0, qy=0.0, qz=-5.0)
        self.assertEqual(ll.distribution, DistributionType.UNIFORM)
        self.assertEqual(ll.qx, 10.0)
        self.assertEqual(ll.qy, 0.0)
        self.assertEqual(ll.qz, -5.0)
        # End values should default to 0.0 for uniform distribution
        self.assertEqual(ll.qx_end, 0.0)
        self.assertEqual(ll.qy_end, 0.0)
        self.assertEqual(ll.qz_end, 0.0)

    def test_linear_load_values(self):
        """Linear LineLoad should store end values for qx, qy, qz."""
        ll = LineLoad("L4", "c", line=self.valid_line, distribution=DistributionType.LINEAR,
                      qx=1.0, qy=2.0, qz=3.0, qx_end=4.0, qy_end=5.0, qz_end=6.0)
        self.assertEqual(ll.distribution, DistributionType.LINEAR)
        self.assertEqual(ll.qx_end, 4.0)
        self.assertEqual(ll.qy_end, 5.0)
        self.assertEqual(ll.qz_end, 6.0)
        # Start values remain as given
        self.assertEqual(ll.qx, 1.0)
        self.assertEqual(ll.qy, 2.0)
        self.assertEqual(ll.qz, 3.0)

    def test_repr_contains_key_info(self):
        """__repr__ for LineLoad should include name, distribution, and stage."""
        ll = LineLoad("LoadLine", "c", line=self.valid_line, qx=0.0, qy=0.0, qz=0.0)
        rep = repr(ll)
        self.assertIn("LineLoad", rep)
        self.assertIn("Static", rep)
        self.assertIn(ll.distribution.name, rep)  # e.g., "UNIFORM"

class TestSurfaceLoad(unittest.TestCase):
    def test_invalid_surface_input(self):
        """SurfaceLoad should require surface to have callable as_tuple_list if attribute exists."""
        class DummySurface:
            as_tuple_list = 123  # has attribute but not callable
        # This should trigger the as_tuple_list validation ValueError
        with self.assertRaises(ValueError):
            SurfaceLoad("S1", "c", surface=DummySurface(), distribution=DistributionType.UNIFORM)

    def test_stress_values_and_gradients(self):
        """SurfaceLoad should store stress components and gradients properly."""
        surf = Polygon3D()
        grads = {"gz_z": -9.81}
        ref = (0.0, 0.0, 0.0)
        sl = SurfaceLoad("S2", "c", surface=surf, distribution=DistributionType.LINEAR,
                         sigmax=1.0, sigmay=2.0, sigmaz=3.0,
                         sigmax_end=4.0, sigmay_end=5.0, sigmaz_end=6.0,
                         gradients=grads, ref_point=ref)
        # Distribution and stage
        self.assertEqual(sl.distribution, DistributionType.LINEAR)
        self.assertEqual(sl.stage, LoadStage.STATIC)
        # Stress values stored in properties
        self.assertEqual(sl.sigmax, 1.0)
        self.assertEqual(sl.sigmay_end, 5.0)
        self.assertEqual(sl.sigmaz_end, 6.0)
        # Gradient dict and reference point should be stored via base class
        self.assertEqual(sl.grad, grads)
        self.assertEqual(sl.ref_point, ref)

    def test_repr_contains_key_info(self):
        """__repr__ for SurfaceLoad should include name, distribution, and stage."""
        surf = Polygon3D()
        sl = SurfaceLoad("S3", "c", surface=surf, distribution=DistributionType.UNIFORM,
                         sigmax=0.0, sigmay=0.0, sigmaz=0.0)
        rep = repr(sl)
        self.assertIn("SurfaceLoad", rep)
        self.assertIn("UNIFORM", rep)
        self.assertIn("Static", rep)

class TestDynamicLoads(unittest.TestCase):
    def setUp(self):
        # Common geometry and multiplier for dynamic load tests
        self.pt = Point(0, 0, 0)
        self.line = Line3D(PointSet([Point(0, 0, 0), Point(1, 0, 0)]))
        self.lm = LoadMultiplier("LM1", "", SignalType.HARMONIC, amplitude=1.0, phase=0.0, frequency=1.0)

    def test_dynpointload_multiplier_handling(self):
        """DynPointLoad should accept multipliers for allowed keys and reject invalid ones."""
        dpl = DynPointLoad("DP1", "c", point=self.pt, Fx=0.0, Fy=0.0, Fz=0.0,
                           multiplier={"Fx": self.lm})
        # The multiplier for 'Fx' should be stored and retrievable
        self.assertIs(dpl.multiplier("Fx"), self.lm)
        # Setting an invalid key should raise ValueError
        with self.assertRaises(ValueError):
            dpl.set_multiplier("invalid", self.lm)
        # Setting a non-LoadMultiplier object should raise TypeError
        with self.assertRaises(TypeError):
            dpl.set_multiplier("Fy", "not_a_multiplier")
        # Setting a new valid multiplier key
        lm2 = LoadMultiplier("LM2", "", SignalType.HARMONIC, amplitude=2.0, phase=0.0, frequency=1.0)
        dpl.set_multiplier("Fy", lm2)
        self.assertIs(dpl.multiplier("Fy"), lm2)

    def test_dynlineload_multiplier_keys(self):
        """DynLineLoad should only accept multipliers for 'qx', 'qy', 'qz' components."""
        dll = DynLineLoad("DL1", "c", line=self.line, multiplier={"qx": self.lm})
        # Allowed key 'qx' should be set
        self.assertIs(dll.multiplier("qx"), self.lm)
        # Not allowed key (e.g., 'Fx') should raise ValueError
        with self.assertRaises(ValueError):
            dll.set_multiplier("Fx", self.lm)
        # _allowed_mul_keys should contain 'qx', 'qy', 'qz' and not 'Fx'
        keys = dll._allowed_mul_keys()
        self.assertIn("qx", keys)
        self.assertIn("qy", keys)
        self.assertIn("qz", keys)
        self.assertNotIn("Fx", keys)

    def test_dynamic_repr_includes_multipliers(self):
        """__repr__ for dynamic loads should include multiplier information."""
        mulX = LoadMultiplier("mX", "", SignalType.HARMONIC, amplitude=1.0, phase=0.0, frequency=1.0)
        mulY = LoadMultiplier("mY", "", SignalType.HARMONIC, amplitude=1.0, phase=0.0, frequency=1.0)
        dp = DynPointLoad("DP2", "c", point=self.pt, Fx=5.0, Fy=5.0, Fz=0.0,
                          multiplier={"Fx": mulX, "Fy": mulY})
        repr_str = repr(dp)
        self.assertIn("DynPointLoad", repr_str)
        self.assertIn("mult=", repr_str)
        # Both 'Fx' and 'Fy' multiplier keys should be mentioned in the repr
        self.assertIn("Fx:", repr_str)
        self.assertIn("Fy:", repr_str)

if __name__ == '__main__':
    unittest.main()
