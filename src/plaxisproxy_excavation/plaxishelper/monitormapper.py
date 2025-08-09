from typing import Any, List
from ..components.curvepoint import CurvePoint

class MonitorMapper:
    """Defines monitoring points (curve points) in the Plaxis model for result extraction."""
    
    @classmethod
    def create_monitors(cls, g_i: Any, monitors: List[CurvePoint]) -> None:
        """
        Creates monitoring points in Plaxis (e.g., for pre-calculation curves).
        
        Args:
            g_i (Any): The Plaxis global input object.
            monitors (List[CurvePoint]): List of curve point objects to define in Plaxis.
        """
        print("Mapping Monitors (Curve Points)...")
        if not monitors:
            return  # No monitors to create
        for point in monitors:
            try:
                # Create a point in Plaxis at the given coordinates
                plaxis_cp = g_i.point(point.x, point.y, point.z)
                point.plx_id = plaxis_cp
                if point.plx_id is not None:
                    print(f"  - Monitor '{point.label}' at ({point.x:.2f}, {point.y:.2f}, {point.z:.2f}) defined for Plaxis.")
                else:
                    print(f"  - ERROR: Monitor '{point.label}' plx_id not assigned!")
            except Exception as e:
                print(f"  - ERROR creating monitor '{point.label}': {e}")
