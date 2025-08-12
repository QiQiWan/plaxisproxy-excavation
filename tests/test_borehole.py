import unittest
from types import SimpleNamespace
from plaxisproxy_excavation.borehole import Borehole, BoreholeSet

class TestBorehole(unittest.TestCase):
    def setUp(self):
        # Create dummy soil material objects with id attributes for testing
        self.soil1 = SimpleNamespace(id="soil1")
        self.soil2 = SimpleNamespace(id="soil2")
        self.soil3 = SimpleNamespace(id="soil3")

    def test_borehole_properties(self):
        """Borehole properties (x, y, h, tables) and repr formatting."""
        bh = Borehole(name="BH1", comment="Test borehole", 
                      x=100.0, y=200.0, h=10.0, 
                      top_list=[1.5, 0.0], bottom_list=[0.0, -5.0], 
                      soil_list=[self.soil1, self.soil2])
        # Coordinate and table properties
        self.assertEqual(bh.x, 100.0)
        self.assertEqual(bh.y, 200.0)
        self.assertEqual(bh.h, 10.0)
        self.assertEqual(bh.top_table, [1.5, 0.0])
        self.assertEqual(bh.bottom_table, [0.0, -5.0])
        self.assertEqual(bh.soil_table, [self.soil1, self.soil2])
        # __repr__ contains id (truncated) and coordinate and layer count
        rep = repr(bh)
        self.assertIn("Borehole", rep)
        self.assertIn("@(100.00, 200.00)", rep)
        self.assertIn(f"n_layers={len(bh.soil_table)}", rep)

    def test_boreholeset_collection(self):
        """BoreholeSet collects unique soils from multiple boreholes."""
        # Borehole 1 with soil1, soil2; Borehole 2 with soil3 and soil1 again
        bh1 = Borehole("BH1", "First", 0,0,0, top_list=[], bottom_list=[], soil_list=[self.soil1, self.soil2])
        bh2 = Borehole("BH2", "Second", 1,1,0, top_list=[], bottom_list=[], soil_list=[self.soil3, self.soil1])
        bh_set = BoreholeSet([bh1, bh2])
        # Borehole list property
        self.assertEqual(bh_set.borehole_list, [bh1, bh2])
        # Soils dictionary should have unique ids
        soils_dict = bh_set.soils
        self.assertEqual(set(soils_dict.keys()), {"soil1", "soil2", "soil3"})
        # Each id maps to the soil object (last one seen)
        self.assertIs(soils_dict["soil1"], self.soil1)
        self.assertIs(soils_dict["soil2"], self.soil2)
        self.assertIs(soils_dict["soil3"], self.soil3)
        # __repr__ reflects number of boreholes and unique soils
        rep = repr(bh_set)
        self.assertIn(f"n_boreholes=2", rep)
        self.assertIn(f"n_unique_soils=3", rep)
