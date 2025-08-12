import unittest
from types import SimpleNamespace
from plaxisproxy_excavation.plaxishelper.plaxismapper import PlaxisMapper

class DummyPlaxisMaterial:
    """Dummy class to simulate a Plaxis material object with arbitrary attributes."""
    def __init__(self):
        self.E = None
        self.nu = None
        self.Name = None

class TestPlaxisMapper(unittest.TestCase):
    def setUp(self):
        # Dummy Plaxis global input (g_i) with required methods
        self.g_i = SimpleNamespace()
        # Provide dummy implementations for needed g_i methods
        self.g_i.setunits = lambda length, force, time: setattr(self.g_i, "units_set", (length, force, time))
        self.g_i.setdomain = lambda x_min, y_min, x_max, y_max: setattr(self.g_i, "domain_set", (x_min, y_min, x_max, y_max))
        self.g_i.soilmat = lambda: DummyPlaxisMaterial()
        self.g_i.platemat = lambda: DummyPlaxisMaterial()
        self.g_i.borehole = lambda x, y: SimpleNamespace(Head=None)
        self.g_i.soillayer = lambda bh, top: None
        self.g_i.point = lambda x, y, z=None: SimpleNamespace(x=x, y=y, z=z)
        self.g_i.polyline = lambda points: SimpleNamespace(points=points)
        self.g_i.polygon = lambda pline: SimpleNamespace(surface_from=pline)
        # For simplicity, no actual plate/beam methods needed in this test
    
    def test_set_project_information(self):
        """Project information mapping calls setunits and setdomain with correct values."""
        mapper = PlaxisMapper(self.g_i)
        proj_info = SimpleNamespace(title="Proj", length_unit=SimpleNamespace(value="m"), 
                                    internal_force_unit=SimpleNamespace(value="kN"), time_unit=SimpleNamespace(value="day"),
                                    x_min=0.0, y_min=0.0, x_max=100.0, y_max=50.0)
        mapper.set_project_information(proj_info)
        # Check that g_i.setunits was called with the Enum .value attributes and setdomain with boundaries
        self.assertEqual(self.g_i.units_set, ("m", "kN", "day"))
        self.assertEqual(self.g_i.domain_set, (0.0, 0.0, 100.0, 50.0))

    def test_create_materials_mapping(self):
        """Materials mapping creates Plaxis materials and assigns attributes."""
        mapper = PlaxisMapper(self.g_i)
        # Dummy material with to_dict returning E and nu
        mat = SimpleNamespace(name="TestMat", to_dict=lambda: {"E": 100.0, "nu": 0.3, "extra": 42})
        materials = {"soil_materials": [mat]}
        mapper.create_materials(materials)
        # Should have created one soilmat and assigned attributes
        # The dummy Plaxis material instance created via soilmat should be stored in object_map
        self.assertIn("TestMat", mapper.object_map)
        plx_mat = mapper.object_map["TestMat"]
        # Attributes E and nu should be set on the plx_mat (existing attributes)
        self.assertEqual(plx_mat.E, 100.0)
        self.assertEqual(plx_mat.nu, 0.3)
        # Non-matching attribute "extra" not set on DummyPlaxisMaterial
        self.assertFalse(hasattr(plx_mat, "extra"))

    def test_create_boreholes_and_geometry_integration(self):
        """Boreholes mapping within PlaxisMapper uses GeometryMapper logic and sets layers."""
        mapper = PlaxisMapper(self.g_i)
        # Create dummy boreholes similar to GeometryMapper test
        bh1 = SimpleNamespace(name="BH1", x=0.0, y=0.0, h=5.0, top_table=[-1.0, -2.0], plx_id=None)
        bh_set = SimpleNamespace(borehole_list=[bh1])
        mapper.create_boreholes_and_geometry(bh_set)
        # After mapping, borehole plx_id should be set and soillayer called for each top
        self.assertIsNotNone(bh1.plx_id)
        # The dummy g_i.soillayer does nothing, but ensure no exceptions and head is set
        self.assertAlmostEqual(bh1.plx_id.Head, 5.0)

    def test_get_plaxis_mat_command(self):
        """_get_plaxis_mat_command returns correct command or raises for unknown category."""
        mapper = PlaxisMapper(self.g_i)
        self.assertEqual(mapper._get_plaxis_mat_command("soil_materials"), "soilmat")
        self.assertEqual(mapper._get_plaxis_mat_command("beam_materials"), "beammat")
        with self.assertRaises(ValueError):
            mapper._get_plaxis_mat_command("invalid_category")
