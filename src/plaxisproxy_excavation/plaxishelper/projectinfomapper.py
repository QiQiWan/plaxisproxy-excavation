from typing import Any
from ..components.projectinformation import ProjectInformation

class ProjectInfoMapper:
    """Handles mapping of project information (units, boundaries) to the Plaxis model."""
    
    @classmethod
    def set_project_information(cls, g_i: Any, proj_info: ProjectInformation) -> None:
        """
        Sets global project settings like units and model boundaries in Plaxis.
        
        Args:
            g_i (Any): The Plaxis global input object (e.g., g_i).
            proj_info (ProjectInformation): The project information data container.
        """
        print("Mapping Project Information...")
        # Set the units (length, force, time) as defined in the project info
        g_i.setunits(
            proj_info.length_unit.value,
            proj_info.internal_force_unit.value,
            proj_info.time_unit.value
        )
        # Set the model domain boundaries (xmin, ymin, xmax, ymax)
        g_i.setdomain(proj_info.x_min, proj_info.y_min, proj_info.x_max, proj_info.y_max)
        print(f"  - Project '{proj_info.title}' settings applied.")
