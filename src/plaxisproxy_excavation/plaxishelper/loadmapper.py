from typing import Any, List, Dict
from ..structures.load import _BaseLoad, PointLoad, LineLoad, SurfaceLoad, DistributionType
from structuremapper import StructureMapper

class LoadMapper:
    """Maps load objects (point, line, surface loads) from the model to Plaxis load entities."""
    
    @classmethod
    def create_loads(cls, g_i: Any, loads: Dict[str, List[_BaseLoad]]) -> None:
        """
        Creates all loads in Plaxis according to the model's load definitions.
        
        Args:
            g_i (Any): The Plaxis global input object.
            loads (Dict[str, List[_BaseLoad]]): Dictionary of load lists categorized by load type.
        """
        print("Mapping Loads...")
        for load_type, load_list in loads.items():
            for load in load_list:
                plaxis_load = None
                try:
                    # Point load
                    if isinstance(load, PointLoad):
                        plaxis_load = g_i.pointload(
                            g_i.point(load.point.x, load.point.y, load.point.z),
                            "CreatePointLoad", load.name
                        )
                        # Set point load forces and moments
                        plaxis_load.Fx = load.Fx; plaxis_load.Fy = load.Fy; plaxis_load.Fz = load.Fz
                        plaxis_load.Mx = load.Mx; plaxis_load.My = load.My; plaxis_load.Mz = load.Mz
                    # Line load
                    elif isinstance(load, LineLoad):
                        points = [g_i.point(p.x, p.y, p.z) for p in load.line.get_points()]
                        plaxis_line = g_i.polyline(points)
                        plaxis_load = g_i.lineload(plaxis_line, "CreateLineLoad", load.name)
                        # Set line load distributed forces
                        plaxis_load.qx = load.qx; plaxis_load.qy = load.qy; plaxis_load.qz = load.qz
                        if load.distribution == DistributionType.LINEAR:
                            # For linear distribution, set end values
                            plaxis_load.qx_end = load.qx_end; plaxis_load.qy_end = load.qy_end; plaxis_load.qz_end = load.qz_end
                    # Surface load
                    elif isinstance(load, SurfaceLoad):
                        plaxis_surface = StructureMapper._create_plaxis_surface(g_i, load.surface)
                        plaxis_load = g_i.surfaceload(plaxis_surface, "CreateSurfaceLoad", load.name)
                        # Set surface load stress components (assuming uniform distribution)
                        plaxis_load.sigmax = load.sigmax; plaxis_load.sigmay = load.sigmay; plaxis_load.sigmaz = load.sigmaz
                    # Assign the Plaxis load reference and log success if created
                    if plaxis_load:
                        load.plx_id = plaxis_load
                        if load.plx_id is not None:
                            print(f"  - Load '{load.name}' of type '{type(load).__name__}' created.")
                        else:
                            print(f"  - ERROR: Load '{load.name}' plx_id not assigned!")
                except Exception as e:
                    print(f"  - ERROR creating load '{load.name}': {e}")
