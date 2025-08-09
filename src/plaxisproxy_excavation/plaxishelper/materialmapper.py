from typing import Any, List, Dict, Optional
from ..core.plaxisobject import PlaxisObject

class MaterialMapper:
    """Maps material definitions from the Python model to Plaxis material datasets."""
    
    @classmethod
    def create_materials(cls, g_i: Any, materials: Dict[str, List[Any]]) -> None:
        """
        Creates material datasets in Plaxis for all materials in the model.
        
        Args:
            g_i (Any): The Plaxis global input object.
            materials (Dict[str, List[Any]]): Dictionary of material lists categorized by type.
        """
        print("Mapping Materials...")
        for mat_type, mat_list in materials.items():
            if not mat_list:
                continue  # Skip if no materials of this type
            for py_mat in mat_list:
                try:
                    plaxis_mat_command = cls._get_plaxis_mat_command(mat_type)
                    # Create a new material in Plaxis using the appropriate command
                    plaxis_mat = getattr(g_i, plaxis_mat_command)()
                    # Map attributes from the Python material object to the Plaxis material object.
                    # Assumes Python attribute names match Plaxis API property names.
                    for attr, value in py_mat.to_dict().items():
                        if hasattr(plaxis_mat, attr):
                            setattr(plaxis_mat, attr, value)
                    # Store the Plaxis object reference in the model object's plx_id
                    py_mat.plx_id = plaxis_mat
                    # Existence check for plx_id
                    if py_mat.plx_id is not None:
                        print(f"  - Material '{py_mat.name}' created in Plaxis.")
                    else:
                        print(f"  - ERROR: Material '{py_mat.name}' plx_id not assigned!")
                except Exception as e:
                    print(f"  - ERROR creating material '{py_mat.name}': {e}")
    
    @staticmethod
    def _get_plaxis_mat_command(mat_category: str) -> str:
        """
        Helper to get the Plaxis API command for creating a material dataset, based on category.
        
        Args:
            mat_category (str): The category of material (e.g., 'soil_materials', 'plate_materials').
        
        Returns:
            str: The corresponding Plaxis creation command name.
        
        Raises:
            ValueError: If the material category is unrecognized.
        """
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
