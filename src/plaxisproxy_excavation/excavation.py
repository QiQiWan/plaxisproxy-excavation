import json
import uuid
from typing import List, Dict, Any, Type, TypeVar, Sequence

# --- Core & Components ---
from .core.plaxisobject import SerializableBase, PlaxisObject
from .components.projectinformation import ProjectInformation
from .components.phase import Phase
from .components.curvepoint import CurvePoint, NodePoint, StressPoint # Import curve point objects
from .borehole import BoreholeSet

# --- Structures ---
from .structures.retainingwall import RetainingWall
from .structures.anchor import Anchor
from .structures.beam import Beam
from .structures.well import Well
from .structures.embeddedpile import EmbeddedPile

# --- Loads ---
from .structures.load import (
    PointLoad, LineLoad, SurfaceLoad, LoadMultiplier,
    DynPointLoad, DynLineLoad, DynSurfaceLoad, UniformSurfaceLoad,
    YAlignedIncrementSurfaceLoad,
    ZAlignedIncrementSurfaceLoad, VectorAlignedIncrementSurfaceLoad,
    FreeIncrementSurfaceLoad, PerpendicularSurfaceLoad
)

# --- Materials ---
from .materials.soilmaterial import SoilMaterialFactory, SoilMaterialsType
from .materials.platematerial import ElasticPlate, ElastoplasticPlate
from .materials.anchormaterial import ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor
from .materials.beammaterial import ElasticBeam, ElastoplasticBeam
from .materials.pilematerial import ElasticPile, ElastoplasticPile


# A generic TypeVar to properly type hint the from_dict class method.
T = TypeVar('T', bound='FoundationPit')

class FoundationPit(SerializableBase):
    """
    Represents a complete Plaxis 3D foundation pit engineering model.
    This object encapsulates all the necessary information for a numerical simulation analysis
    of a foundation pit and provides functionality to save the entire model to and
    load it from a JSON file.
    """
    # Using __slots__ for memory optimization and faster attribute access.
    __slots__ = (
        "_id",
        "_version",
        "project_information",
        "borehole_set",
        "materials",
        "structures",
        "loads",
        "phases",
        "monitors" # For curve point
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
        self.borehole_set: BoreholeSet = BoreholeSet(boreholes=[])
        
        # Material library: managed in categories using a dictionary
        self.materials: Dict[str, List[Any]] = {
            "soil_materials": [],
            "plate_materials": [],
            "anchor_materials": [],
            "beam_materials": [],
            "pile_materials": []
        }
        
        # Structure library
        self.structures: Dict[str, List[Any]] = {
            "retaining_walls": [],
            "anchors": [],
            "beams": [],
            "wells": [],
            "embedded_piles": [],
            "soil_blocks": []
        }

        # Load library
        self.loads: Dict[str, List[Any]] = {
            "point_loads": [],
            "line_loads": [],
            "surface_loads": []
        }

        # Construction phases
        self.phases: List[Phase] = []

        # Monitors: for storing curve points to be monitored
        self.monitors: List[CurvePoint] = []

    def _is_duplicate(self, new_obj: PlaxisObject, existing_list: Sequence[PlaxisObject]) -> bool:
        """
        Checks if an object with the same name already exists in a list.
        All objects (materials, structures, loads, phases) inherit from PlaxisObject
        and have a 'name' attribute which should be unique within its category.
        """
        # Check if any object in the list has the same name.
        return any(existing.name == new_obj.name for existing in existing_list)

    def add_material(self, material_type: str, material_obj: Any):
        """
        Adds a material definition to the model, preventing duplicates.

        Args:
            material_type (str): The type of the material (e.g., 'soil_materials').
            material_obj (Any): The instance of the material object.
        
        Raises:
            ValueError: If the material type is unsupported or a material with the same name already exists.
        """
        if material_type not in self.materials:
            raise ValueError(f"Unsupported material type: {material_type}")
        
        if self._is_duplicate(material_obj, self.materials[material_type]):
            print(f"Warning: Material '{material_obj.name}' of type '{material_type}' already exists. Skipping.")
            return

        self.materials[material_type].append(material_obj)

    def add_structure(self, structure_type: str, structure_obj: Any):
        """
        Adds a structural component to the model, preventing duplicates.

        Args:
            structure_type (str): The type of the structure (e.g., 'retaining_walls').
            structure_obj (Any): The instance of the structure object.
            
        Raises:
            ValueError: If the structure type is unsupported or a structure with the same name already exists.
        """
        if structure_type not in self.structures:
            raise ValueError(f"Unsupported structure type: {structure_type}")

        if self._is_duplicate(structure_obj, self.structures[structure_type]):
            print(f"Warning: Structure '{structure_obj.name}' of type '{structure_type}' already exists. Skipping.")
            return

        self.structures[structure_type].append(structure_obj)

    def add_load(self, load_type: str, load_obj: Any):
        """
        Adds a load to the model, preventing duplicates.

        Args:
            load_type (str): The type of the load (e.g., 'point_loads').
            load_obj (Any): The instance of the load object.
            
        Raises:
            ValueError: If the load type is unsupported or a load with the same name already exists.
        """
        if load_type not in self.loads:
            raise ValueError(f"Unsupported load type: {load_type}")
            
        if self._is_duplicate(load_obj, self.loads[load_type]):
            print(f"Warning: Load '{load_obj.name}' of type '{load_type}' already exists. Skipping.")
            return

        self.loads[load_type].append(load_obj)

    def add_phase(self, phase: Phase):
        """
        Adds a construction stage to the model, preventing duplicates.

        Args:
            phase (ConstructionStage): The construction stage object.
            
        Raises:
            ValueError: If a phase with the same name already exists.
        """
        if self._is_duplicate(phase, self.phases):
            print(f"Warning: Phase '{phase.name}' already exists. Skipping.")
            return
            
        self.phases.append(phase)
        
    def add_monitor_point(self, point: CurvePoint):
        """
        Adds a monitoring curve point to the model, preventing duplicates.
        Uniqueness is based on the object instance itself or its unique label.

        Args:
            point (CurvePoint): The curve point object to monitor.
        """
        # Check for the exact same object instance
        if point in self.monitors:
            print(f"Warning: The exact monitor point object is already in the list. Skipping.")
            return
        
        # If the point has a label, check for duplicate labels
        if point.label and any(p.label == point.label for p in self.monitors):
            print(f"Warning: A monitor point with the label '{point.label}' already exists. Skipping.")
            return
            
        self.monitors.append(point)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the PlaxisFoundationPit object to a dictionary.
        Leverages the _serialize_value method from the SerializableBase parent class
        for recursive serialization.
        """
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "_id": self._id,
            "_version": self._version,
            "project_information": self._serialize_value(self.project_information),
            "borehole_set": self._serialize_value(self.borehole_set),
            "materials": self._serialize_value(self.materials),
            "structures": self._serialize_value(self.structures),
            "loads": self._serialize_value(self.loads),
            "phases": self._serialize_value(self.phases),
            "monitors": self._serialize_value(self.monitors)
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Deserializes a dictionary to create a PlaxisFoundationPit object.
        This method is specifically tailored to handle the polymorphism within the model.
        """
        # --- 1. Define Type Mappings ---
        # This makes the deserialization logic cleaner and more extensible.
        material_mapping = {
            # Soil materials are handled by the factory; this maps other materials.
            "plate_materials": {
                "ElasticPlate": ElasticPlate,
                "ElastoplasticPlate": ElastoplasticPlate
            },
            "anchor_materials": {
                "ElasticAnchor": ElasticAnchor,
                "ElastoplasticAnchor": ElastoplasticAnchor,
                "ElastoPlasticResidualAnchor": ElastoPlasticResidualAnchor
            },
            "beam_materials": {
                "ElasticBeam": ElasticBeam,
                "ElastoplasticBeam": ElastoplasticBeam
            },
            "pile_materials": {
                "ElasticPile": ElasticPile,
                "ElastoplasticPile": ElastoplasticPile
            }
        }
        structure_mapping = {
            "retaining_walls": RetainingWall,
            "anchors": Anchor,
            "beams": Beam,
            "wells": Well,
            "embedded_piles": EmbeddedPile
        }
        load_mapping = {
            "point_loads": PointLoad,
            "line_loads": LineLoad,
            "surface_loads": SurfaceLoad
        }
        monitor_mapping = {
            "NodePoint": NodePoint,
            "StressPoint": StressPoint,
            "CurvePoint": CurvePoint # Fallback
        }

        # --- 2. Instantiate the main object ---
        # First, create ProjectInformation
        proj_info_data = data["project_information"]
        instance = cls(ProjectInformation.from_dict(proj_info_data))
        # Restore metadata
        instance._id = data["_id"]
        instance._version = data.get("_version", "1.0.0")

        # --- 3. Deserialize nested objects ---
        if "borehole_set" in data:
            instance.borehole_set = BoreholeSet.from_dict(data["borehole_set"])

        # Deserialize materials using the defined mapping
        for m_type, m_list in data.get("materials", {}).items():
            if m_type == "soil_materials":
                # Use the factory to correctly instantiate different soil models
                instance.materials[m_type] = [
                    SoilMaterialFactory.create(SoilMaterialsType[item['type']], **item) 
                    for item in m_list
                ]
            elif m_type in material_mapping:
                # For other materials, dynamically select the class based on the __type__ field
                cls_map = material_mapping[m_type]
                deserialized_list = []
                for item in m_list:
                    class_name = item.get("__type__", "").split('.')[-1]
                    if class_name in cls_map:
                        deserialized_list.append(cls_map[class_name].from_dict(item))
                instance.materials[m_type] = deserialized_list

        # Deserialize structures using the defined mapping
        for s_type, s_list in data.get("structures", {}).items():
            if s_type in structure_mapping:
                cls_to_use = structure_mapping[s_type]
                instance.structures[s_type] = [cls_to_use.from_dict(item) for item in s_list]

        # Deserialize loads using the defined mapping
        for l_type, l_list in data.get("loads", {}).items():
            deserialized_list = []
            for item in l_list:
                class_name = item.get("__type__", "").split('.')[-1]
                if class_name in load_mapping:
                    deserialized_list.append(load_mapping[class_name].from_dict(item))
            instance.loads[l_type] = deserialized_list
        
        # Deserialize construction phases
        if "phases" in data:
            instance.phases = [Phase.from_dict(p) for p in data["phases"]]
            
        # Deserialize monitor points
        if "monitors" in data:
            deserialized_monitors = []
            for item in data["monitors"]:
                 class_name = item.get("__type__", "").split('.')[-1]
                 if class_name in monitor_mapping:
                     deserialized_monitors.append(monitor_mapping[class_name].from_dict(item))
            instance.monitors = deserialized_monitors

        return instance

    def save_to_json(self, file_path: str):
        """
        Saves the entire model object to a JSON file.

        Args:
            file_path (str): The path to the file where the model will be saved.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4, default=str)
        print(f"Model successfully saved to: {file_path}")

    @classmethod
    def load_from_json(cls: Type[T], file_path: str) -> T:
        """
        Loads a model object from a JSON file.

        Args:
            file_path (str): The path to the file from which to load the model.

        Returns:
            PlaxisFoundationPit: The instantiated PlaxisFoundationPit object loaded from the file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Logic to handle different versions could be added here
        # For example: if data.get('_version') == '1.0.0': ...
        
        instance = cls.from_dict(data)
        print(f"Model successfully loaded from {file_path}.")
        return instance