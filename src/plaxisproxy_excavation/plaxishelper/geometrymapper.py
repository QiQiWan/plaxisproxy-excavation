from typing import Any, List
from ..borehole import Borehole, BoreholeSet

class GeometryMapper:
    """Creates boreholes and defines soil stratigraphy (ground geometry) in Plaxis."""
    
    @classmethod
    def create_boreholes_and_geometry(cls, g_i: Any, borehole_set: BoreholeSet) -> None:
        """
        Creates boreholes in Plaxis according to the borehole set and adds soil layers.
        
        Args:
            g_i (Any): The Plaxis global input object.
            borehole_set (BoreholeSet): Collection of borehole data to be created.
        """
        print("Mapping Boreholes and Soil Geometry...")
        if not borehole_set.borehole_list:
            return  # No boreholes to map
        for bh in borehole_set.borehole_list:
            # Create a borehole at the specified coordinates
            plaxis_bh = g_i.borehole(bh.x, bh.y)
            plaxis_bh.Head = bh.h  # Set the borehole head (ground surface elevation)
            # Assign the Plaxis borehole reference to the model object
            bh.plx_id = plaxis_bh
            # Existence check
            if bh.plx_id is None:
                print(f"  - ERROR: Borehole '{bh.name}' plx_id not assigned!")
            # Add each soil layer from the borehole definition into Plaxis
            for top_level in bh.top_table:
                g_i.soillayer(plaxis_bh, top_level)
            print(f"  - Borehole '{bh.name}' geometry defined at ({bh.x}, {bh.y}).")
        print("  - Assigning materials to generated soil volumes...")
        # After borehole creation, Plaxis auto-generates soil volumes for each layer.
        # The following is a placeholder for assigning materials to these soil volumes.
        # A possible approach is to iterate through g_i.Soils and match volumes to borehole soil materials.
        # Example (pseudo-code): g_i.Soils[0].Material = <material_reference>
        print("    - NOTE: Material assignment to soil volumes needs to be implemented based on Plaxis's object naming.")
