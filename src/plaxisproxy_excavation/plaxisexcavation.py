import json
from typing import List, Dict, Any, Type, TypeVar
import uuid

from .core.plaxisobject import PlaxisObject, SerializableBase
from .components.projectinformation import ProjectInformation
from .borehole import BoreholeSet
from .structures.retainingwall import RetainingWall
from .structures.anchor import Anchor
from .structures.beam import Beam
from .structures.well import Well
from .structures.load import PointLoad, LineLoad, SurfaceLoad
from .components.phase import ConstructionStage
from .materials.soilmaterial import BaseSoilMaterial, SoilMaterialFactory, SoilMaterialsType
from .materials.platematerial import ElasticPlate
from .materials.anchormaterial import ElasticAnchor
from .materials.beammaterial import ElasticBeam
from .geometry import Point, Line3D, PointSet, Polygon3D

# A generic TypeVar to properly type hint the from_dict class method.
T = TypeVar('T', bound='SerializableBase')

class PlaxisFoundationPit(SerializableBase):
    """
    Represents a complete Plaxis 3D foundation pit engineering model.
    This object encapsulates all the necessary information for a numerical simulation analysis
    of a foundation pit and provides functionality to save the entire model to and
    load it from a JSON file.
    """
    # Using __slots__ for memory optimization and faster attribute access.
    __slots__ = (
        "_id",
        "project_information",
        "borehole_set",
        "structures",
        "loads",
        "phases",
        "materials",
        "_version"  # Added for future-proofing
    )

    def __init__(self, project_information: ProjectInformation):
        """
        Initializes the PlaxisFoundationPit object.

        Args:
            project_information (ProjectInformation): Basic information about the project.
        """
        self._id = str(uuid.uuid4())
        self._version = "1.0.0"  # Semantic versioning for the data model
        self.project_information = project_information
        self.borehole_set: BoreholeSet = BoreholeSet([])
        self.structures: Dict[str, List[Any]] = {
            "retaining_walls": [],
            "anchors": [],
            "beams": [],
            "wells": [],
            "embedded_piles": []
        }
        self.loads: Dict[str, List[Any]] = {
            "point_loads": [],
            "line_loads": [],
            "surface_loads": []
        }
        self.phases: List[ConstructionStage] = []
        self.materials: Dict[str, List[Any]] = {
            "soil_materials": [],
            "plate_materials": [],
            "anchor_materials": [],
            "beam_materials": [],
            "pile_materials": []
        }

    def add_structure(self, structure_type: str, structure_obj: Any):
        """
        Adds a structural component to the model.

        Args:
            structure_type (str): The type of the structure (e.g., 'retaining_walls', 'anchors').
            structure_obj (Any): The instance of the structure object.
        """
        if structure_type in self.structures:
            self.structures[structure_type].append(structure_obj)
        else:
            raise ValueError(f"Unsupported structure type: {structure_type}")

    def add_load(self, load_type: str, load_obj: Any):
        """
        Adds a load to the model.

        Args:
            load_type (str): The type of the load (e.g., 'point_loads', 'line_loads').
            load_obj (Any): The instance of the load object.
        """
        if load_type in self.loads:
            self.loads[load_type].append(load_obj)
        else:
            raise ValueError(f"Unsupported load type: {load_type}")

    def add_phase(self, phase: ConstructionStage):
        """
        Adds a construction stage to the model.

        Args:
            phase (ConstructionStage): The construction stage object.
        """
        self.phases.append(phase)

    def add_material(self, material_type: str, material_obj: Any):
        """
        Adds a material definition to the model.

        Args:
            material_type (str): The type of the material (e.g., 'soil_materials').
            material_obj (Any): The instance of the material object.
        """
        if material_type in self.materials:
            self.materials[material_type].append(material_obj)
        else:
            raise ValueError(f"Unsupported material type: {material_type}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the PlaxisFoundationPit object to a dictionary.
        Leverages the _serialize_value method from the SerializableBase parent class
        for recursive serialization.
        """
        return {
            "_id": self._id,
            "_version": self._version,
            "project_information": self._serialize_value(self.project_information),
            "borehole_set": self._serialize_value(self.borehole_set),
            "structures": self._serialize_value(self.structures),
            "loads": self._serialize_value(self.loads),
            "phases": self._serialize_value(self.phases),
            "materials": self._serialize_value(self.materials)
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Deserializes a dictionary to create a PlaxisFoundationPit object.
        This method is specifically tailored to handle the polymorphism within the model.
        """
        # --- Pre-computation and Mapping Setup ---
        # This makes the deserialization logic cleaner and more extensible.
        structure_mapping = {
            "retaining_walls": RetainingWall,
            "anchors": Anchor,
            "beams": Beam,
            "wells": Well,
        }
        load_mapping = {
            "point_loads": PointLoad,
            "line_loads": LineLoad,
            "surface_loads": SurfaceLoad
        }

        # --- Instantiate the main object ---
        # First, create ProjectInformation, which may need special handling
        proj_info_data = data["project_information"]
        # Assuming ProjectInformation is not a SerializableBase and needs manual instantiation
        from .components.projectinformation import Units
        proj_info_data['length_unit'] = Units.Length(proj_info_data['length_unit'])
        proj_info_data['internal_force_unit'] = Units.Force(proj_info_data['internal_force_unit'])
        proj_info_data['time_unit'] = Units.Time(proj_info_data['time_unit'])
        proj_info_data.pop('id', None) # 'id' is internal, not a constructor argument
        instance = cls(ProjectInformation(**proj_info_data))

        # Restore metadata
        instance._id = data["_id"]
        instance._version = data.get("_version", "1.0.0") # Handle older files without version

        # --- Deserialize nested objects ---
        instance.borehole_set = BoreholeSet.from_dict(data["borehole_set"])

        # Deserialize structures using the defined mapping
        for s_type, s_list in data["structures"].items():
            if s_type in structure_mapping:
                cls_to_use = structure_mapping[s_type]
                instance.structures[s_type] = [cls_to_use.from_dict(item) for item in s_list]

        # Deserialize loads using the defined mapping
        for l_type, l_list in data["loads"].items():
            if l_type in load_mapping:
                cls_to_use = load_mapping[l_type]
                instance.loads[l_type] = [cls_to_use.from_dict(item) for item in l_list]
        
        # Deserialize construction phases
        instance.phases = [ConstructionStage.from_dict(p) for p in data["phases"]]

        # Deserialize materials, with special handling for the SoilMaterialFactory
        for m_type, m_list in data["materials"].items():
            if m_type == "soil_materials":
                # Use the factory to correctly instantiate different soil models (MC, HSS, etc.)
                instance.materials[m_type] = [SoilMaterialFactory.create(SoilMaterialsType[item['type']], **item) for item in m_list]
            else:
                # A more generic approach for other material types
                material_cls_map = {
                    "plate_materials": ElasticPlate,
                    "anchor_materials": ElasticAnchor,
                    "beam_materials": ElasticBeam,
                }
                if m_type in material_cls_map:
                    cls_to_use = material_cls_map[m_type]
                    instance.materials[m_type] = [cls_to_use.from_dict(item) for item in m_list]
        
        return instance

    def save_to_json(self, file_path: str):
        """
        Saves the entire model object to a JSON file.

        Args:
            file_path (str): The path to the file where the model will be saved.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        print(f"Model successfully saved to: {file_path}")

    @classmethod
    def load_from_json(cls, file_path: str) -> "PlaxisFoundationPit":
        """
        Loads a model object from a JSON file.

        Args:
            file_path (str): The path to the file from which to load the model.

        Returns:
            PlaxisFoundationPit: The instantiated PlaxisFoundationPit object loaded from the file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Here you could add logic to handle different versions if needed
        # For example: if data.get('_version') == '1.0.0': ...
        
        instance = cls.from_dict(data)
        print(f"Model successfully loaded from {file_path}.")
        return instance