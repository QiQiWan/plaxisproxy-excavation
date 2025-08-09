from typing import Any, List, Optional
from ..geometry import Polygon3D
from ..core.plaxisobject import PlaxisObject

class StructureMapper:
    """Maps structural elements (walls, anchors, beams, piles, wells) from the model to Plaxis geometry."""
    
    @classmethod
    def create_structures(cls, g_i: Any, structures: Dict[str, List[Any]]) -> None:
        """
        Creates structural elements in Plaxis (plates, anchors, beams, etc.) according to the model data.
        
        Args:
            g_i (Any): The Plaxis global input object.
            structures (Dict[str, List[Any]]): Dictionary of structure lists categorized by type.
        """
        print("Mapping Structures...")
        # Retaining walls (modeled as plates)
        for wall in structures.get("retaining_walls", []):
            plaxis_surface = cls._create_plaxis_surface(g_i, wall.surface)
            plaxis_plate = g_i.plate(plaxis_surface, "CreatePlate", wall.name)
            cls._assign_material(plaxis_plate, getattr(wall, "plate_type", None))
            wall.plx_id = plaxis_plate
            if wall.plx_id is None:
                print(f"  - ERROR: Retaining Wall '{wall.name}' plx_id not assigned!")
            else:
                print(f"  - Retaining Wall (Plate) '{wall.name}' created.")
        # Anchors (node-to-node anchors or fixed-end anchors)
        for anchor in structures.get("anchors", []):
            p1, p2 = anchor.line.get_points()
            # The specific Plaxis command may differ based on anchor type (e.g., fixed end vs node-to-node)
            plaxis_anchor = g_i.nodetonodeanchor(g_i.point(p1.x, p1.y, p1.z),
                                                 g_i.point(p2.x, p2.y, p2.z))
            plaxis_anchor.Name = anchor.name
            cls._assign_material(plaxis_anchor, getattr(anchor, "anchor_type", None))
            anchor.plx_id = plaxis_anchor
            if anchor.plx_id is None:
                print(f"  - ERROR: Anchor '{anchor.name}' plx_id not assigned!")
            else:
                print(f"  - Anchor '{anchor.name}' created.")
        # Beams (embedded beam elements along polylines)
        for beam in structures.get("beams", []):
            points = [g_i.point(p.x, p.y, p.z) for p in beam.line.get_points()]
            plaxis_line = g_i.polyline(points)
            plaxis_beam = g_i.beam(plaxis_line, "CreateBeam", beam.name)
            cls._assign_material(plaxis_beam, getattr(beam, "beam_type", None))
            beam.plx_id = plaxis_beam
            if beam.plx_id is None:
                print(f"  - ERROR: Beam '{beam.name}' plx_id not assigned!")
            else:
                print(f"  - Beam '{beam.name}' created.")
        # Embedded piles (modeled as embedded beam rows)
        for pile in structures.get("embedded_piles", []):
            points = [g_i.point(p.x, p.y, p.z) for p in pile.line.get_points()]
            plaxis_line = g_i.polyline(points)
            plaxis_pile = g_i.embeddedbeamrow(plaxis_line)
            plaxis_pile.Name = pile.name
            cls._assign_material(plaxis_pile, getattr(pile, "pile_type", None))
            pile.plx_id = plaxis_pile
            if pile.plx_id is None:
                print(f"  - ERROR: Embedded Pile '{pile.name}' plx_id not assigned!")
            else:
                print(f"  - Embedded Pile '{pile.name}' created.")
        # Wells (well elements along a line, with type and settings)
        for well in structures.get("wells", []):
            points = [g_i.point(p.x, p.y, p.z) for p in well.line.get_points()]
            plaxis_line = g_i.polyline(points)
            plaxis_well = g_i.well(plaxis_line)
            plaxis_well.Name = well.name
            plaxis_well.Type = getattr(well.well_type, "value", well.well_type)  # e.g., "Extraction"
            plaxis_well.h_min = well.h_min
            well.plx_id = plaxis_well
            if well.plx_id is None:
                print(f"  - ERROR: Well '{well.name}' plx_id not assigned!")
            else:
                print(f"  - Well '{well.name}' created.")
    
    @staticmethod
    def _create_plaxis_surface(g_i: Any, py_surface: Polygon3D) -> Any:
        """
        Helper to create a Plaxis surface from a Polygon3D object.
        
        Args:
            g_i (Any): Plaxis global input object.
            py_surface (Polygon3D): The polygon defining the surface.
        
        Returns:
            Any: The Plaxis surface object created.
        """
        outer_ring_points = [g_i.point(p.x, p.y, p.z) for p in py_surface.outer_ring.get_points()]
        plaxis_polyline = g_i.polyline(outer_ring_points)
        plaxis_surface = g_i.polygon(plaxis_polyline)
        # TODO: Handle inner rings (holes) if needed in the future.
        return plaxis_surface
    
    @staticmethod
    def _assign_material(plaxis_obj: Any, py_material: Optional[PlaxisObject]) -> None:
        """
        Helper to assign a material to a Plaxis object by linking via the material's plx_id.
        
        Args:
            plaxis_obj (Any): The Plaxis object (structure) to assign material to.
            py_material (Optional[PlaxisObject]): The Python material object to assign (or None).
        """
        if py_material and getattr(py_material, "plx_id", None):
            # Assign the material reference if available
            plaxis_obj.Material = py_material.plx_id
        else:
            material_name = py_material.name if py_material else "None"
            obj_name = getattr(plaxis_obj, "Name", "<unnamed>")
            print(f"  - WARNING: Material '{material_name}' for object '{obj_name}' not found or not created.")
