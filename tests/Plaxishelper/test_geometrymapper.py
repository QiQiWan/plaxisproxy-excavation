import unittest
from types import SimpleNamespace
from plaxisproxy_excavation.plaxishelper.geometrymapper import GeometryMapper

class TestGeometryMapper(unittest.TestCase):
    def test_create_boreholes_and_geometry(self):
        """GeometryMapper.create_boreholes_and_geometry maps boreholes and layers using g_i API."""
        # Dummy global input with borehole() and soillayer() methods
        created_bh_calls = []
        soillayer_calls = []
        def dummy_borehole(x, y):
            created_bh_calls.append((x, y))
            # Return a dummy Plaxis borehole object with Head attribute
            return SimpleNamespace(Head=None)
        dummy_g_i = SimpleNamespace(
            borehole=lambda x, y: dummy_borehole(x, y),
            soillayer=lambda bh, top: soillayer_calls.append((bh, top))
        )
        # Prepare BoreholeSet with two boreholes
        BH = SimpleNamespace  # alias for brevity
        borehole1 = BH(name="BH1", x=1.0, y=2.0, h=10.0, top_table=[5.0, 0.0], plx_id=None)
        borehole2 = BH(name="BH2", x=3.0, y=4.0, h=15.0, top_table=[7.0], plx_id=None)
        borehole_set = BH(borehole_list=[borehole1, borehole2])
        # Call the mapper
        GeometryMapper.create_boreholes_and_geometry(dummy_g_i, borehole_set)
        # It should call g_i.borehole for each borehole
        self.assertEqual(created_bh_calls, [(1.0, 2.0), (3.0, 4.0)])
        # Each borehole's plx_id should be set to the returned borehole object
        self.assertIsNotNone(borehole1.plx_id)
        self.assertIsNotNone(borehole2.plx_id)
        # Each returned Plaxis borehole's Head set to borehole.h
        self.assertAlmostEqual(borehole1.plx_id.Head, 10.0)
        self.assertAlmostEqual(borehole2.plx_id.Head, 15.0)
        # soillayer should be called for each top level in each borehole
        expected_soil_calls = len(borehole1.top_table) + len(borehole2.top_table)
        self.assertEqual(len(soillayer_calls), expected_soil_calls)
        # Check that each call used the correct Plaxis borehole reference and top value
        tops_seen = {call[1] for call in soillayer_calls}
        self.assertEqual(tops_seen, {5.0, 0.0, 7.0})
