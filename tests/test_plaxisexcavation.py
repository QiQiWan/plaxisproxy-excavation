import io, json, os
import unittest
from types import SimpleNamespace
from plaxisproxy_excavation.plaxisexcavation import PlaxisFoundationPit

class TestPlaxisFoundationPit(unittest.TestCase):
    def setUp(self):
        # Dummy project_information with minimal to_dict
        self.proj_info = SimpleNamespace(to_dict=lambda: {"title": "TestProject"})

    def test_initial_state(self):
        """PlaxisFoundationPit initializes with empty collections and correct keys."""
        pf = PlaxisFoundationPit(self.proj_info)
        # Unique id and version
        self.assertIsInstance(pf._id, str)
        self.assertEqual(pf._version, "1.0.0")
        # Project info assigned
        self.assertEqual(pf.project_information.to_dict(), {"title": "TestProject"})
        # Borehole set default is empty
        self.assertEqual(pf.borehole_set.borehole_list, [])
        # Materials, structures, loads initialized with correct keys and empty lists
        expected_mat_keys = {"soil_materials", "plate_materials", "anchor_materials", "beam_materials", "pile_materials"}
        self.assertEqual(set(pf.materials.keys()), expected_mat_keys)
        for lst in pf.materials.values():
            self.assertEqual(lst, [])
        expected_struct_keys = {"retaining_walls", "anchors", "beams", "wells", "embedded_piles"}
        self.assertEqual(set(pf.structures.keys()), expected_struct_keys)
        for lst in pf.structures.values():
            self.assertEqual(lst, [])
        expected_load_keys = {"point_loads", "line_loads", "surface_loads"}
        self.assertEqual(set(pf.loads.keys()), expected_load_keys)
        for lst in pf.loads.values():
            self.assertEqual(lst, [])
        self.assertEqual(pf.phases, [])
        self.assertEqual(pf.monitors, [])

    def test_is_duplicate_helper(self):
        """_is_duplicate returns True if an object with same name exists in list."""
        pf = PlaxisFoundationPit(self.proj_info)
        dummy1 = SimpleNamespace(name="X")
        dummy2 = SimpleNamespace(name="Y")
        existing_list = [dummy1]
        # Same name as dummy1 -> duplicate
        self.assertTrue(pf._is_duplicate(SimpleNamespace(name="X"), existing_list))
        # Different name -> not duplicate
        self.assertFalse(pf._is_duplicate(dummy2, existing_list))

    def test_add_material(self):
        """add_material adds new materials and skips duplicates or invalid types."""
        pf = PlaxisFoundationPit(self.proj_info)
        matA = SimpleNamespace(name="MatA")
        matB = SimpleNamespace(name="MatB")
        # Unsupported material type should raise
        with self.assertRaises(ValueError):
            pf.add_material("unknown_type", matA)
        # Add first material
        pf.add_material("soil_materials", matA)
        self.assertIn(matA, pf.materials["soil_materials"])
        # Duplicate name (different object with same name) should be skipped (list unchanged)
        matA_copy = SimpleNamespace(name="MatA")
        pf.add_material("soil_materials", matA_copy)
        self.assertEqual(len(pf.materials["soil_materials"]), 1)
        # Add another distinct material
        pf.add_material("soil_materials", matB)
        self.assertEqual(len(pf.materials["soil_materials"]), 2)
        self.assertIn(matB, pf.materials["soil_materials"])

    def test_add_structure(self):
        """add_structure adds new structures and skips duplicates or invalid types."""
        pf = PlaxisFoundationPit(self.proj_info)
        struct1 = SimpleNamespace(name="Wall1")
        struct2 = SimpleNamespace(name="Wall2")
        with self.assertRaises(ValueError):
            pf.add_structure("invalid_type", struct1)
        pf.add_structure("retaining_walls", struct1)
        self.assertIn(struct1, pf.structures["retaining_walls"])
        # Duplicate by name
        struct1_dup = SimpleNamespace(name="Wall1")
        pf.add_structure("retaining_walls", struct1_dup)
        self.assertEqual(len(pf.structures["retaining_walls"]), 1)
        pf.add_structure("retaining_walls", struct2)
        self.assertEqual(len(pf.structures["retaining_walls"]), 2)
        self.assertIn(struct2, pf.structures["retaining_walls"])

    def test_add_load(self):
        """add_load adds new loads and skips duplicates or invalid types."""
        pf = PlaxisFoundationPit(self.proj_info)
        load1 = SimpleNamespace(name="Load1")
        load2 = SimpleNamespace(name="Load2")
        with self.assertRaises(ValueError):
            pf.add_load("unsupported_type", load1)
        pf.add_load("point_loads", load1)
        self.assertIn(load1, pf.loads["point_loads"])
        # Duplicate skip
        load1_dup = SimpleNamespace(name="Load1")
        pf.add_load("point_loads", load1_dup)
        self.assertEqual(len(pf.loads["point_loads"]), 1)
        pf.add_load("point_loads", load2)
        self.assertEqual(len(pf.loads["point_loads"]), 2)
        self.assertIn(load2, pf.loads["point_loads"])

    def test_add_phase_and_monitor(self):
        """add_phase and add_monitor_point handle duplicates appropriately."""
        pf = PlaxisFoundationPit(self.proj_info)
        phase1 = SimpleNamespace(name="Phase1")
        phase2 = SimpleNamespace(name="Phase1")  # same name as phase1
        pf.add_phase(phase1)
        self.assertIn(phase1, pf.phases)
        pf.add_phase(phase2)  # duplicate name should be skipped
        self.assertEqual(len(pf.phases), 1)
        # Monitor points
        pt1 = SimpleNamespace(label="M1")
        pt2 = SimpleNamespace(label="M1")  # different object, same label
        pf.add_monitor_point(pt1)
        self.assertIn(pt1, pf.monitors)
        # Adding the same object again - skip
        pf.add_monitor_point(pt1)
        self.assertEqual(len(pf.monitors), 1)
        # Adding different object with same label - skip
        pf.add_monitor_point(pt2)
        self.assertEqual(len(pf.monitors), 1)
        # Adding a new point with different label
        pt3 = SimpleNamespace(label="M2")
        pf.add_monitor_point(pt3)
        self.assertEqual(len(pf.monitors), 2)
        self.assertIn(pt3, pf.monitors)

    def test_to_dict_serialization(self):
        """to_dict produces a serializable dictionary covering all components."""
        pf = PlaxisFoundationPit(self.proj_info)
        # Set up one dummy in each category
        pf.materials["soil_materials"].append(SimpleNamespace(name="SoilX", to_dict=lambda: {"name": "SoilX"}))
        pf.structures["anchors"].append(SimpleNamespace(name="AnchorX", to_dict=lambda: {"name": "AnchorX"}))
        pf.loads["point_loads"].append(SimpleNamespace(name="LoadX", to_dict=lambda: {"name": "LoadX"}))
        pf.phases.append(SimpleNamespace(name="PhaseX", to_dict=lambda: {"name": "PhaseX"}))
        pf.monitors.append(SimpleNamespace(x=0, y=0, z=0, label="Mon1", to_dict=lambda: {"label": "Mon1"}))
        d = pf.to_dict()
        # Check presence of top-level keys
        self.assertIn("__type__", d)
        self.assertIn("_id", d)
        self.assertIn("project_information", d)
        self.assertIn("materials", d)
        self.assertIn("structures", d)
        self.assertIn("loads", d)
        self.assertIn("phases", d)
        self.assertIn("monitors", d)
        # Materials, structures, loads should be dictionaries of lists
        self.assertIn("soil_materials", d["materials"])
        self.assertIn("anchors", d["structures"])
        self.assertIn("point_loads", d["loads"])
        # The inserted dummy objects should appear in serialization
        mat_list = d["materials"]["soil_materials"]
        self.assertTrue(any(item.get("name") == "SoilX" for item in mat_list))
        struct_list = d["structures"]["anchors"]
        self.assertTrue(any(item.get("name") == "AnchorX" for item in struct_list))
        load_list = d["loads"]["point_loads"]
        self.assertTrue(any(item.get("name") == "LoadX" for item in load_list))
        phase_list = d["phases"]
        self.assertTrue(any(item.get("name") == "PhaseX" for item in phase_list))
        monitor_list = d["monitors"]
        self.assertTrue(any(item.get("label") == "Mon1" for item in monitor_list))
        # JSON serializable check (no errors on dumping)
        try:
            json_str = json.dumps(d)
        except TypeError as e:
            self.fail(f"PlaxisFoundationPit to_dict output not JSON serializable: {e}")

    def test_save_and_load_json(self):
        """save_to_json writes file and load_from_json reconstructs PlaxisFoundationPit."""
        pf = PlaxisFoundationPit(self.proj_info)
        pf.project_information = SimpleNamespace(to_dict=lambda: {"title": "Project A"}, **{"from_dict": staticmethod(lambda d: SimpleNamespace(**d))})
        # Use a temporary file path
        file_path = "temp_test_model.json"
        # Ensure no exception during save
        pf.save_to_json(file_path)
        self.assertTrue(os.path.exists(file_path))
        # Load the file back
        loaded_pf = PlaxisFoundationPit.load_from_json(file_path)
        # Basic check: loaded object is PlaxisFoundationPit with matching project title
        self.assertIsInstance(loaded_pf, PlaxisFoundationPit)
        self.assertEqual(getattr(loaded_pf.project_information, "title", ""), "Project A")
        # Clean up temp file
        os.remove(file_path)
