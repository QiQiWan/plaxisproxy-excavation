# src/plaxisproxy_excavation/plaxismapper.py

from typing import List, Dict, Any, Optional

# --- Import all necessary objects from the library ---
# This ensures the mapper is aware of all types it needs to handle.

# Core & Components
from ..core.plaxisobject import PlaxisObject
from ..plaxisexcavation import PlaxisFoundationPit
from ..borehole import Borehole, BoreholeSet
from ..components.phase import ConstructionStage
from ..components.curvepoint import CurvePoint, NodePoint, StressPoint
from ..components.projectinformation import ProjectInformation

# Geometry
from ..geometry import Point, Line3D, Polygon3D

# Structures
from ..structures.retainingwall import RetainingWall
from ..structures.anchor import Anchor
from ..structures.beam import Beam
from ..structures.well import Well
from ..structures.embeddedpile import EmbeddedPile

# Loads - Import all specific load types for type checking
from ..structures.load import (
    _BaseLoad, PointLoad, LineLoad, SurfaceLoad, LoadMultiplier,
    DistributionType
)

# Materials - Import all specific material types for type checking
from ..materials.soilmaterial import BaseSoilMaterial
from ..materials.platematerial import ElasticPlate, ElastoplasticPlate
from ..materials.anchormaterial import ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor
from ..materials.beammaterial import ElasticBeam, ElastoplasticBeam
from ..materials.pilematerial import ElasticPile, ElastoplasticPile


class PlaxisMapper:
    """
    Translates a PlaxisFoundationPit object into a series of Plaxis API commands.
    
    This class acts as the core "translator" between the user-friendly Python object
    model and the Plaxis remote scripting interface. It iterates through the model's
    components and generates the corresponding Plaxis objects and settings.
    """

    def __init__(self, g_i: Any):
        """
        Initializes the mapper with a reference to the Plaxis global input object.

        Args:
            g_i (Any): The global object from the Plaxis remote scripting server (e.g., g_i).
        """
        self.g_i = g_i
        # A dictionary to map the Python object's name to the created Plaxis object reference.
        # This is crucial for linking objects, e.g., assigning a material to a structure.
        self.object_map: Dict[str, Any] = {}

    def run(self, model: PlaxisFoundationPit):
        """
        Executes the full mapping process to create the entire model in Plaxis.
        
        The order of operations is critical and follows Plaxis's logical workflow:
        1. Set project-level information.
        2. Create material datasets.
        3. Define soil stratigraphy and geometry.
        4. Create structural elements.
        5. Apply loads.
        6. Define construction phases and activities.
        7. Set up monitoring points.
        """
        print("--- Starting Plaxis Model Generation ---")
        self.set_project_information(model.project_information)
        self.create_materials(model.materials)
        self.create_boreholes_and_geometry(model.borehole_set)
        self.create_structures(model.structures)
        self.create_loads(model.loads)
        self.create_phases(model.phases)
        self.create_monitors(model.monitors)
        print("--- Plaxis Model Generation Complete ---")

    def set_project_information(self, proj_info: ProjectInformation):
        """Sets global project settings like units and model boundaries."""
        print("Mapping Project Information...")
        self.g_i.setunits(
            proj_info.length_unit.value,
            proj_info.internal_force_unit.value,
            proj_info.time_unit.value
        )
        self.g_i.setdomain(proj_info.x_min, proj_info.y_min, proj_info.x_max, proj_info.y_max)
        print(f"  - Project '{proj_info.title}' settings applied.")

    def create_materials(self, materials: Dict[str, List[Any]]):
        """Maps all material objects to Plaxis material datasets."""
        print("Mapping Materials...")
        for mat_type, mat_list in materials.items():
            if not mat_list:
                continue
            
            for py_mat in mat_list:
                try:
                    plaxis_mat_command = self._get_plaxis_mat_command(mat_type)
                    plaxis_mat = getattr(self.g_i, plaxis_mat_command)()
                    
                    # Automatically map attributes from the Python object to the Plaxis object.
                    # This relies on the convention that Python attribute names match Plaxis API properties.
                    for attr, value in py_mat.to_dict().items():
                        if hasattr(plaxis_mat, attr):
                            setattr(plaxis_mat, attr, value)
                    
                    self.object_map[py_mat.name] = plaxis_mat
                    print(f"  - Material '{py_mat.name}' created in Plaxis.")
                except Exception as e:
                    print(f"  - ERROR creating material '{py_mat.name}': {e}")

    def create_boreholes_and_geometry(self, borehole_set: BoreholeSet):
        """Creates boreholes and defines the soil stratigraphy."""
        print("Mapping Boreholes and Soil Geometry...")
        if not borehole_set.borehole_list:
            return
            
        for bh in borehole_set.borehole_list:
            plaxis_bh = self.g_i.borehole(bh.x, bh.y)
            plaxis_bh.Head = bh.h # Set the borehole head level
            
            for i, top_level in enumerate(bh.top_table):
                # Add a soil layer at a certain level.
                self.g_i.soillayer(plaxis_bh, top_level)
            
            print(f"  - Borehole '{bh.name}' geometry defined at ({bh.x}, {bh.y}).")

        print("  - Assigning materials to generated soil volumes...")
        # After defining borehole geometries, Plaxis generates soil volumes.
        # This part requires a strategy to link Python soil objects to Plaxis soil volumes.
        # A common method is to iterate through the auto-generated soil objects in Plaxis
        # and assign materials based on the borehole definition.
        # The exact command depends on the Plaxis version and can be complex.
        # For example: `g_i.Soils[0].Material = self.object_map['Sand']`
        # This is a placeholder for the user to implement based on their specific logic.
        print("    - NOTE: Material assignment to soil volumes needs to be implemented based on Plaxis's object naming.")


    def create_structures(self, structures: Dict[str, List[Any]]):
        """Maps all structure objects to Plaxis geometry and structural elements."""
        print("Mapping Structures...")
        
        for wall in structures.get("retaining_walls", []):
            plaxis_surface = self._create_plaxis_surface(wall.surface)
            plaxis_plate = self.g_i.plate(plaxis_surface, "CreatePlate", wall.name)
            self._assign_material(plaxis_plate, wall.plate_type)
            self.object_map[wall.name] = plaxis_plate
            print(f"  - Retaining Wall (Plate) '{wall.name}' created.")

        for anchor in structures.get("anchors", []):
            p1, p2 = anchor.line.get_points()
            # The command may differ based on anchor type (e.g., nodetonodeanchor, fixedendanchor)
            plaxis_anchor = self.g_i.nodetonodeanchor(self.g_i.point(p1.x, p1.y, p1.z), self.g_i.point(p2.x, p2.y, p2.z))
            plaxis_anchor.Name = anchor.name
            self._assign_material(plaxis_anchor, anchor.anchor_type)
            self.object_map[anchor.name] = plaxis_anchor
            print(f"  - Anchor '{anchor.name}' created.")

        for beam in structures.get("beams", []):
            points = [self.g_i.point(p.x, p.y, p.z) for p in beam.line.get_points()]
            plaxis_line = self.g_i.polyline(points)
            plaxis_beam = self.g_i.beam(plaxis_line, "CreateBeam", beam.name)
            self._assign_material(plaxis_beam, beam.beam_type)
            self.object_map[beam.name] = plaxis_beam
            print(f"  - Beam '{beam.name}' created.")

        for pile in structures.get("embedded_piles", []):
            points = [self.g_i.point(p.x, p.y, p.z) for p in pile.line.get_points()]
            plaxis_line = self.g_i.polyline(points)
            # The command in Plaxis is typically 'embeddedbeamrow'
            plaxis_pile = self.g_i.embeddedbeamrow(plaxis_line)
            plaxis_pile.Name = pile.name
            self._assign_material(plaxis_pile, pile.pile_type)
            self.object_map[pile.name] = plaxis_pile
            print(f"  - Embedded Pile '{pile.name}' created.")

        for well in structures.get("wells", []):
            points = [self.g_i.point(p.x, p.y, p.z) for p in well.line.get_points()]
            plaxis_line = self.g_i.polyline(points)
            plaxis_well = self.g_i.well(plaxis_line)
            plaxis_well.Name = well.name
            plaxis_well.Type = well.well_type.value # e.g., "Extraction"
            plaxis_well.h_min = well.h_min
            self.object_map[well.name] = plaxis_well
            print(f"  - Well '{well.name}' created.")

    def create_loads(self, loads: Dict[str, List[_BaseLoad]]):
        """Maps all load objects to Plaxis loads, handling different types and distributions."""
        print("Mapping Loads...")
        for load_type, load_list in loads.items():
            for load in load_list:
                plaxis_load = None
                try:
                    if isinstance(load, PointLoad):
                        plaxis_load = self.g_i.pointload(self.g_i.point(load.point.x, load.point.y, load.point.z), "CreatePointLoad", load.name)
                        plaxis_load.Fx, plaxis_load.Fy, plaxis_load.Fz = load.Fx, load.Fy, load.Fz
                        plaxis_load.Mx, plaxis_load.My, plaxis_load.Mz = load.Mx, load.My, load.Mz
                    elif isinstance(load, LineLoad):
                        points = [self.g_i.point(p.x, p.y, p.z) for p in load.line.get_points()]
                        plaxis_line = self.g_i.polyline(points)
                        plaxis_load = self.g_i.lineload(plaxis_line, "CreateLineLoad", load.name)
                        plaxis_load.qx, plaxis_load.qy, plaxis_load.qz = load.qx, load.qy, load.qz
                        if load.distribution == DistributionType.LINEAR:
                            plaxis_load.qx_end, plaxis_load.qy_end, plaxis_load.qz_end = load.qx_end, load.qy_end, load.qz_end
                    elif isinstance(load, SurfaceLoad):
                        plaxis_surface = self._create_plaxis_surface(load.surface)
                        plaxis_load = self.g_i.surfaceload(plaxis_surface, "CreateSurfaceLoad", load.name)
                        # This requires complex mapping for different distributions
                        plaxis_load.sigmax, plaxis_load.sigmay, plaxis_load.sigmaz = load.sigmax, load.sigmay, load.sigmaz
                    
                    if plaxis_load:
                        self.object_map[load.name] = plaxis_load
                        print(f"  - Load '{load.name}' of type '{type(load).__name__}' created.")
                except Exception as e:
                    print(f"  - ERROR creating load '{load.name}': {e}")

    def create_phases(self, phases: List[ConstructionStage]):
        """Creates and configures construction phases, including object activation/deactivation."""
        print("Mapping Construction Phases...")
        if not phases:
            return
            
        previous_plaxis_phase = self.g_i.Phases[0] # Initial Phase

        for py_phase in phases:
            new_plaxis_phase = self.g_i.phase(previous_plaxis_phase)
            new_plaxis_phase.Name = py_phase.name
            
            # Map calculation settings from the Python object to the Plaxis phase
            for setting_name, setting_value in py_phase.settings.settings.items():
                if hasattr(new_plaxis_phase, setting_name):
                    setattr(new_plaxis_phase, setting_name, setting_value)
            
            self._set_phase_activity(new_plaxis_phase, py_phase._activate, True)
            self._set_phase_activity(new_plaxis_phase, py_phase._deactivate, False)
            
            for soil_block in py_phase._excavate:
                # Excavation is handled by deactivating soil volumes
                if soil_block.name in self.object_map:
                    target_soil_volume = self.object_map[soil_block.name]
                    target_soil_volume.Active = False
                    print(f"    - Excavation: Deactivated '{soil_block.name}' in phase '{py_phase.name}'.")
                else:
                    # In Plaxis, soil volumes are often named automatically (e.g., Soil_1, Soil_2)
                    # A robust implementation needs to map these names.
                    print(f"    - WARNING: Could not find soil block '{soil_block.name}' to excavate. It might have a different name in Plaxis.")

            previous_plaxis_phase = new_plaxis_phase
            print(f"  - Phase '{py_phase.name}' created.")

    def create_monitors(self, monitors: List[CurvePoint]):
        """Creates monitoring points (Curve Points) for result extraction."""
        print("Mapping Monitors (Curve Points)...")
        if not monitors:
            return
        
        for point in monitors:
            try:
                # The command in Plaxis is g_i.precalc() for pre-calculation points
                # or defined in Output for post-calculation curves.
                # This assumes we are defining points for pre-calculation.
                plaxis_cp = self.g_i.point(point.x, point.y, point.z)
                # The command to select a node for curves might be different, e.g., g_i.selectnode(...)
                print(f"  - Monitor '{point.label}' at ({point.x:.2f}, {point.y:.2f}, {point.z:.2f}) defined for Plaxis.")
            except Exception as e:
                print(f"  - ERROR creating monitor '{point.label}': {e}")

    # --- Private Helper Methods ---

    def _create_plaxis_surface(self, py_surface: Polygon3D) -> Any:
        """Helper to create a Plaxis surface from a Polygon3D object."""
        outer_ring_points = [self.g_i.point(p.x, p.y, p.z) for p in py_surface.outer_ring.get_points()]
        plaxis_poly_line = self.g_i.polyline(outer_ring_points)
        plaxis_surface = self.g_i.polygon(plaxis_poly_line)
        # To-do: Add logic for inner rings (holes) if necessary
        return plaxis_surface

    def _assign_material(self, plaxis_obj: Any, py_material: Optional[PlaxisObject]):
        """Helper to assign a Python material object to a Plaxis object by name."""
        if py_material and py_material.name in self.object_map:
            plaxis_obj.Material = self.object_map[py_material.name]
        else:
            material_name = py_material.name if py_material else "None"
            print(f"  - WARNING: Material '{material_name}' for object '{plaxis_obj.Name}' not found in created materials map.")

    def _get_plaxis_mat_command(self, mat_category: str) -> str:
        """Helper to get the Plaxis command string for creating a material dataset."""
        mapping = {
            "soil_materials": "soilmat",
            "plate_materials": "platemat",
            "anchor_materials": "anchormat",
            "beam_materials": "beammat",
            "pile_materials": "pilemat",
            "load_multipliers": "loadmultiplier",
        }
        if mat_category not in mapping:
            raise ValueError(f"Unknown material category for Plaxis command mapping: {mat_category}")
        return mapping[mat_category]
        
    def _set_phase_activity(self, plaxis_phase: Any, object_list: List[Any], active_state: bool):
        """Helper to activate or deactivate a list of objects in a given phase."""
        action_str = "Activating" if active_state else "Deactivating"
        for item in object_list:
            obj_name = item.name if isinstance(item, PlaxisObject) else str(item)
            if obj_name in self.object_map:
                # In Plaxis scripting, you often access the object's state within the phase context
                target_obj_in_phase = getattr(plaxis_phase, self.object_map[obj_name].Name)
                target_obj_in_phase.Active = active_state
            else:
                print(f"    - WARNING: Cannot set activity for '{obj_name}' in phase '{plaxis_phase.Name}', object not found in map.")
