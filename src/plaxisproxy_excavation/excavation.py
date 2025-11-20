import json
import uuid
from typing import List, Dict, Any, Type, TypeVar, Sequence

# ### Core & Components ###
from .core.plaxisobject import SerializableBase, PlaxisObject
from .components.projectinformation import ProjectInformation
from .components.phase import Phase
from .components.curvepoint import CurvePoint, NodePoint, StressPoint # Import curve point objects
from .borehole import BoreholeSet

# ### Structures ###
from .structures.retainingwall import RetainingWall
from .structures.anchor import Anchor
from .structures.beam import Beam
from .structures.well import Well
from .structures.embeddedpile import EmbeddedPile

# ### Loads ###
from .structures.load import (
    PointLoad, LineLoad, SurfaceLoad, LoadMultiplier,
    DynPointLoad, DynLineLoad, DynSurfaceLoad, UniformSurfaceLoad,
    YAlignedIncrementSurfaceLoad,
    ZAlignedIncrementSurfaceLoad, VectorAlignedIncrementSurfaceLoad,
    FreeIncrementSurfaceLoad, PerpendicularSurfaceLoad
)

# ### Materials ###
from .materials.soilmaterial import SoilMaterialFactory, SoilMaterialsType
from .materials.platematerial import ElasticPlate, ElastoplasticPlate
from .materials.anchormaterial import ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor
from .materials.beammaterial import ElasticBeam, ElastoplasticBeam
from .materials.pilematerial import ElasticPile, ElastoplasticPile

from typing import List, Tuple, Optional, Any
from plaxisproxy_excavation.structures.retainingwall import RetainingWall
from plaxisproxy_excavation.geometry import Polygon3D, Point, PointSet
from enum import Enum



# A generic TypeVar to properly type hint the from_dict class method.
T = TypeVar('T', bound='FoundationPit')

# ### StructureType Enum and helpers ###
class StructureType(str, Enum):
    """
    Canonical structure bucket names used throughout the project.
    Values are set to the existing string keys to保持完全兼容.
    """
    RETAINING_WALLS = "retaining_walls"
    ANCHORS = "anchors"
    BEAMS = "beams"
    WELLS = "wells"
    EMBEDDED_PILES = "embedded_piles"
    SOIL_BLOCKS = "soil_blocks"

# Allowed canonical keys for quick membership checks
_STRUCTURE_ALLOWED = {t.value for t in StructureType}

# Common synonyms → canonical key (kept small and conservative for compatibility)
_STRUCTURE_SYNONYMS = {
    "retainingwall": StructureType.RETAINING_WALLS.value,
    "retainingwalls": StructureType.RETAINING_WALLS.value,
    "walls": StructureType.RETAINING_WALLS.value,
    "wall": StructureType.RETAINING_WALLS.value,
    "plates": StructureType.RETAINING_WALLS.value,
    "plate": StructureType.RETAINING_WALLS.value,

    "anchor": StructureType.ANCHORS.value,
    "nodetonodeanchor": StructureType.ANCHORS.value,
    "nodetonodeanchors": StructureType.ANCHORS.value,

    "beam": StructureType.BEAMS.value,

    "well": StructureType.WELLS.value,

    "embeddedpile": StructureType.EMBEDDED_PILES.value,
    "embeddedpiles": StructureType.EMBEDDED_PILES.value,

    "soil": StructureType.SOIL_BLOCKS.value,
    "soils": StructureType.SOIL_BLOCKS.value,
    "soilblocks": StructureType.SOIL_BLOCKS.value,
}

def _normalize_structure_type(structure_type: Any) -> str:
    """
    Normalize structure type input (Enum or str) to a canonical string key.
    - Accepts StructureType or string.
    - Returns one of StructureType.*.value when possible.
    - Falls back to the cleaned input string if unknown (with a warning).
    """
    if isinstance(structure_type, StructureType):
        return structure_type.value
    if isinstance(structure_type, str):
        key = structure_type.strip().lower().replace("-", "_").replace(" ", "_")
        if key in _STRUCTURE_ALLOWED:
            return key
        if key in _STRUCTURE_SYNONYMS:
            return _STRUCTURE_SYNONYMS[key]
        # Keep compatibility but surface a gentle warning for typos.
        print(f"Warning: Unknown structure type '{structure_type}'. Using '{key}' as-is.")
        return key
    raise TypeError("structure_type must be a StructureType or str")

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
        "_excava_depth",
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
        self._excava_depth = 0.0
        # Material library: managed in categories using a dictionary
        self.materials: Dict[str, List[Any]] = {
            "soil_materials": [],
            "plate_materials": [],
            "anchor_materials": [],
            "beam_materials": [],
            "pile_materials": []
        }
        
        # Structure library
        self.structures: Dict[str, List[Any]] = {t.value: [] for t in StructureType}

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

    @property
    def excava_depth(self):
        return self._excava_depth

    @excava_depth.setter
    def excava_depth(self, value):
        if value > 0:
            value = -value
        print(f"[INFO] Set {value} m as the depth of the excavation.")
        self._excava_depth = value

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

    def add_structure(self, structure_type: Any, structure_obj: Any):
        """
        Adds a structural component to the model, preventing duplicates.

        Args:
            structure_type (Any): The type of the structure (e.g., 'retaining_walls' or StructureType).
            structure_obj (Any): The instance of the structure object.
        """
        key = _normalize_structure_type(structure_type)
        if key not in self.structures:
            # Keep forward compatibility with custom buckets while encouraging canonical keys
            self.structures[key] = []

        if self._is_duplicate(structure_obj, self.structures[key]):
            print(f"Warning: Structure '{structure_obj.name}' of type '{key}' already exists. Skipping.")
            return

        self.structures[key].append(structure_obj)

    def update_structures(self, structure_type: Any, structure_list: List):
        """
        Update a structural component to the model, substitute the original structures. All phases will be reset to [], after update the structures.

        Args:
            structure_type (Any): The type of the structure (e.g., 'retaining_walls' or StructureType).
            structure_list (Any): The list of the structure objects.
        """
        key = _normalize_structure_type(structure_type)
        self.structures[key] = structure_list


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

    def clear_all_phases(self):
        """
        Delete all of the phases in this object.
        """
        self.phases = []

    # ################# Footprint helpers: walls → closed polygon #################

    @staticmethod
    def _extract_surface_points(surface: Any) -> List[Any]:
        pts = []
        # 常见实现：outer_ring.get_points()
        outer = getattr(surface, "outer_ring", None)
        if outer is not None and hasattr(outer, "get_points"):
            try:
                pts = list(outer.get_points())
            except Exception:
                pts = []
        # 备选：直接 points / point_set.points
        if not pts:
            raw = getattr(surface, "points", None)
            if raw:
                try:
                    pts = list(raw)
                except Exception:
                    pts = []
        if not pts:
            ps = getattr(surface, "point_set", None) or getattr(surface, "_point_set", None)
            inner = getattr(ps, "points", None) or getattr(ps, "_points", None) if ps else None
            if inner:
                pts = list(inner)
        return pts or []

    @staticmethod
    def _extract_wall_top_edge_xy(surface: Any, tol: float) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Extract the projection endpoints (x, y) of the "top edge" on the XY plane from Polygon3D. The value of z_top should be the maximum z value of this plane. 
        First, take the top edges of the adjacent two points; if not found, then revert to the top heights of any two points.
        """
        pts = FoundationPit._extract_surface_points(surface)
        if len(pts) < 2:
            return None
        zmax = max(float(p.z) for p in pts)
        def _is_top(p): return abs(float(p.z) - zmax) <= max(tol, 1e-9)
        # 先找相邻两点构成的顶边
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if _is_top(a) and _is_top(b):
                return ( (float(a.x), float(a.y)), (float(b.x), float(b.y)) )
        # 处理闭合首尾
        a, b = pts[-1], pts[0]
        if _is_top(a) and _is_top(b):
            return ( (float(a.x), float(a.y)), (float(b.x), float(b.y)) )
        # 回退：任取两个顶点
        tops = [p for p in pts if _is_top(p)]
        if len(tops) >= 2:
            a, b = tops[0], tops[1]
            return ( (float(a.x), float(a.y)), (float(b.x), float(b.y)) )
        return None

    @staticmethod
    def _q(v: float, tol: float) -> int:
        return int(round(v / tol))

    @staticmethod
    def _pkey(pt: Tuple[float, float], tol: float) -> Tuple[int, int]:
        return (FoundationPit._q(pt[0], tol), FoundationPit._q(pt[1], tol))

    @staticmethod
    def _stitch_segments_to_loop(
        segs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        tol: float
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Connect several line segments at their endpoints to form a single closed loop. Return the vertices in sequence (without repetition of the beginning and end points).
        """
        if len(segs) < 3:
            return None
        unused = list(range(len(segs)))
        used = set()
        # 选一个确定性起点（字典序最小）
        start_idx = min(unused, key=lambda i: (segs[i][0][0], segs[i][0][1], segs[i][1][0], segs[i][1][1]))
        a, b = segs[start_idx]
        path = [a, b]
        used.add(start_idx)
        curr = b

        def _match_idx(pt):
            k = FoundationPit._pkey(pt, tol)
            for idx in unused:
                if idx in used: 
                    continue
                s0, s1 = segs[idx]
                if FoundationPit._pkey(s0, tol) == k:
                    return idx, False, s1
                if FoundationPit._pkey(s1, tol) == k:
                    return idx, True, s0
            return (-1, -1), False, (-1, -1)

        while len(used) < len(segs):
            idx, rev, nxt = _match_idx(curr)
            if idx is None:
                break
            used.add(idx)
            path.append(nxt)
            curr = nxt
            # 闭合判定
            if FoundationPit._pkey(curr, tol) == FoundationPit._pkey(path[0], tol):
                # 去掉最后一个等于起点的点
                path = path[:-1]
                return [path[0]] + [p for p in path[1:]]
        # 如果自然闭合失败，但已形成 ≥3 点环，尝试首尾拼接
        if len(path) >= 4 and FoundationPit._pkey(path[0], tol) == FoundationPit._pkey(path[-1], tol):
            path = path[:-1]
            return path
        return None

    @staticmethod
    def _compute_wall_footprint_polygon3d_from_walls(
        walls: List[Any],
        z_value: float,
        tol: float = 1e-6
    ) -> Optional[Polygon3D]:
        # 从每面墙抽取顶边段
        segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for w in walls or []:
            surf = getattr(w, "surface", None)
            if surf is None:
                continue
            e = FoundationPit._extract_wall_top_edge_xy(surf, tol)
            if e is not None:
                segs.append(e)
        if len(segs) < 3:
            return None
        # 去重（端点相同即同一段）
        seen = set()
        uniq = []
        for s in segs:
            k0 = (FoundationPit._pkey(s[0], tol), FoundationPit._pkey(s[1], tol))
            k1 = (k0[1], k0[0])
            if k0 in seen or k1 in seen:
                continue
            seen.add(k0); seen.add(k1)
            uniq.append(s)
        loop = FoundationPit._stitch_segments_to_loop(uniq, tol)
        if not loop or len(loop) < 3:
            return None
        pts3d = [Point(x, y, float(z_value)) for (x, y) in loop] + [Point(loop[0][0], loop[0][1], float(z_value))]
        return Polygon3D.from_points(PointSet(pts3d))

    def get_wall_footprint_polygon3d(self, z_value: float = 0.0, tol: float = 1e-6) -> Optional[Polygon3D]:
        walls = []
        if isinstance(getattr(self, "structures", None), dict):
            walls = self.structures.get(StructureType.RETAINING_WALLS.value, []) or []
        if not walls:
            return None
        return FoundationPit._compute_wall_footprint_polygon3d_from_walls(walls, z_value, tol)

    def get_wall_footprint_xy(self, z_value: float = 0.0, tol: float = 1e-6) -> List[Tuple[float, float]]:
        poly = self.get_wall_footprint_polygon3d(z_value=z_value, tol=tol)
        if not poly:
            return []
        pts = []
        if hasattr(poly, "outer_ring") and hasattr(poly.outer_ring, "get_points"):
            pts = list(poly.outer_ring.get_points())
        elif hasattr(poly, "get_points"):
            pts = list(poly.get_points())
        xy = [(float(p.x), float(p.y)) for p in pts]
        if len(xy) >= 2 and xy[0] == xy[-1]:
            xy = xy[:-1]
        return xy

    def get_excavation_bottom_polygon3d(self, tol: float = 1e-6) -> Optional[Polygon3D]:
        z = float(getattr(self, "excava_depth", 0.0))
        return self.get_wall_footprint_polygon3d(z_value=z, tol=tol)


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
        # ### 1. Define Type Mappings ###
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
            StructureType.RETAINING_WALLS.value: RetainingWall,
            StructureType.ANCHORS.value: Anchor,
            StructureType.BEAMS.value: Beam,
            StructureType.WELLS.value: Well,
            StructureType.EMBEDDED_PILES.value: EmbeddedPile
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

        # ### 2. Instantiate the main object ###
        # First, create ProjectInformation
        proj_info_data = data["project_information"]
        instance = cls(ProjectInformation.from_dict(proj_info_data))
        # Restore metadata
        instance._id = data["_id"]
        instance._version = data.get("_version", "1.0.0")

        # ### 3. Deserialize nested objects ###
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
            key_norm = _normalize_structure_type(s_type)
            if key_norm in structure_mapping:
                cls_to_use = structure_mapping[key_norm]
                instance.structures[key_norm] = [cls_to_use.from_dict(item) for item in s_list]

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