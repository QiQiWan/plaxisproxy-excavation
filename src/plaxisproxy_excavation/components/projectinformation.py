from __future__ import annotations
from typing import Dict, Any, Type, TypeVar, Tuple
from enum import Enum
import uuid

__all__ = ["Units", "ProjectInformation"]

T = TypeVar("T", bound="ProjectInformation")


class Units:
    """
    Unit groups used by the project. These enums are intentionally small and
    pragmatic—extend as needed to match your PLAXIS template or corporate standard.
    """
    class Length(Enum):
        MM = "mm"
        CM = "cm"
        M = "m"

    class Force(Enum):
        N = "N"
        KN = "kN"

    class Stress(Enum):
        PA = "Pa"
        KPA = "kPa"
        MPA = "MPa"
        GPA = "GPa"

    class Time(Enum):
        S = "s"
        MIN = "min"
        H = "h"
        DAY = "day"


class ProjectInformation:
    """
    Lightweight, explicit project metadata container.

    Design goals:
      - Be explicit and predictable (no silent mutation of field names).
      - Validate inputs at construction time.
      - Provide a stable serialization format for configs or tests.
      - Offer a `plx_id` field where a PLAXIS-side handle/object can be stored
        by a Mapper after creation.

    Typical usage:
      >>> proj = ProjectInformation(
      ...   title="Basement Excavation", company="ACME", dir="D:/jobs/demo",
      ...   file_name="demo.p3d", comment="pilot run",
      ...   model="3D", element="10-noded",
      ...   length_unit=Units.Length.M, force_unit=Units.Force.KN,
      ...   stress_unit=Units.Stress.KPA, time_unit=Units.Time.DAY,
      ...   gamma_water=9.81, x_min=0, x_max=120, y_min=0, y_max=80
      ... )
      >>> # later: ProjectInformationMapper.create(g_i, proj)  # will write plx_id
    """

    __slots__ = (
        "_id", "_title", "_company", "_dir", "_file_name", "_comment",
        "_model", "_element",
        "_length_unit", "_force_unit", "_stress_unit", "_time_unit",
        "_gamma_water",
        "_x_min", "_x_max", "_y_min", "_y_max",
        "plx_id",
    )

    def __init__(
        self,
        title: str,
        company: str,
        dir: str,
        file_name: str,
        comment: str,
        model: str,
        element: str,
        length_unit: Units.Length,
        force_unit: Units.Force,
        stress_unit: Units.Stress,
        time_unit: Units.Time,
        gamma_water: float,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
    ) -> None:
        # Basic validation to catch configuration errors early.
        if not title:
            raise ValueError("Project title must not be empty.")
        if not dir:
            raise ValueError("Project directory must not be empty.")
        if not file_name:
            raise ValueError("File name must not be empty.")
        if gamma_water <= 0:
            raise ValueError("gamma_water must be positive.")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("(x_max, y_max) must be greater than (x_min, y_min).")

        self._id = uuid.uuid4()
        self._title, self._company = title, company
        self._dir, self._file_name, self._comment = dir, file_name, comment
        self._model, self._element = model, element
        self._length_unit, self._force_unit = length_unit, force_unit
        self._stress_unit, self._time_unit = stress_unit, time_unit
        self._gamma_water = float(gamma_water)
        self._x_min, self._x_max = float(x_min), float(x_max)
        self._y_min, self._y_max = float(y_min), float(y_max)

        # Set by the Mapper after creation in PLAXIS (can be group/handle/object).
        self.plx_id = None

    # --------------------------- read-only properties ---------------------------

    @property
    def id(self) -> uuid.UUID:
        """Opaque unique identifier for this configuration instance."""
        return self._id

    @property
    def title(self) -> str:
        return self._title

    @property
    def company(self) -> str:
        return self._company

    @property
    def dir(self) -> str:
        """Preferred directory for the project (where files are saved)."""
        return self._dir

    @property
    def file_name(self) -> str:
        """Preferred PLAXIS project file name."""
        return self._file_name

    @property
    def comment(self) -> str:
        return self._comment

    @property
    def model(self) -> str:
        """Model type string, e.g., '2D', '3D'."""
        return self._model

    @property
    def element(self) -> str:
        """Element type string, e.g., '15-noded', '10-noded'."""
        return self._element

    @property
    def length_unit(self) -> Units.Length:
        return self._length_unit

    @property
    def force_unit(self) -> Units.Force:
        return self._force_unit

    @property
    def stress_unit(self) -> Units.Stress:
        return self._stress_unit

    @property
    def time_unit(self) -> Units.Time:
        return self._time_unit

    @property
    def gamma_water(self) -> float:
        """Unit weight of water (consistent with chosen units)."""
        return self._gamma_water

    @property
    def x_min(self) -> float:
        return self._x_min

    @property
    def x_max(self) -> float:
        return self._x_max

    @property
    def y_min(self) -> float:
        return self._y_min

    @property
    def y_max(self) -> float:
        return self._y_max

    # ------------------------------ convenience -------------------------------

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Return the project bounding box as (x_min, x_max, y_min, y_max)."""
        return self._x_min, self._x_max, self._y_min, self._y_max

    # ------------------------------- serialization ----------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the project information to a plain dict (JSON-friendly).
        Enums are stored via `.value` to keep the payload clean and robust.
        """
        return {
            "id": str(self._id),
            "title": self._title,
            "company": self._company,
            "dir": self._dir,
            "file_name": self._file_name,
            "comment": self._comment,
            "model": self._model,
            "element": self._element,
            "length_unit": self._length_unit.value,
            "force_unit": self._force_unit.value,
            "stress_unit": self._stress_unit.value,
            "time_unit": self._time_unit.value,
            "gamma_water": self._gamma_water,
            "x_min": self._x_min,
            "x_max": self._x_max,
            "y_min": self._y_min,
            "y_max": self._y_max,
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create a ProjectInformation instance from a dict produced by `to_dict()`.
        Accepts a few legacy/alternate field names to smooth migration.
        """

        def _parse_enum(val: Any, enum_cls: Any, default: Any) -> Any:
            if val is None:
                return default
            if isinstance(val, enum_cls):
                return val
            if isinstance(val, str):
                s = val.strip()
                # Accept both member name and value
                try:
                    if s in enum_cls.__members__:
                        return enum_cls[s]
                except Exception:
                    pass
                try:
                    return enum_cls(s)
                except Exception:
                    return default
            try:
                return enum_cls(val)
            except Exception:
                return default

        # Legacy aliasing for directory
        dir_val = data.get("dir") or data.get("project_dir") or data.get("directory") or "."

        length_unit = _parse_enum(data.get("length_unit"), Units.Length, Units.Length.M)
        force_unit = _parse_enum(data.get("force_unit"), Units.Force, Units.Force.KN)
        stress_unit = _parse_enum(data.get("stress_unit"), Units.Stress, Units.Stress.KPA)
        time_unit = _parse_enum(data.get("time_unit"), Units.Time, Units.Time.DAY)

        inst = cls(
            title=data.get("title", "Untitled"),
            company=data.get("company", "Unknown"),
            dir=dir_val,
            file_name=data.get("file_name", "project.p3d"),
            comment=data.get("comment", ""),
            model=data.get("model", "3D"),
            element=data.get("element", "10-noded"),
            length_unit=length_unit,
            force_unit=force_unit,
            stress_unit=stress_unit,
            time_unit=time_unit,
            gamma_water=float(data.get("gamma_water", 9.81)),
            x_min=float(data.get("x_min", 0.0)),
            x_max=float(data.get("x_max", 100.0)),
            y_min=float(data.get("y_min", 0.0)),
            y_max=float(data.get("y_max", 100.0)),
        )
        # Preserve incoming ID if present (best-effort).
        try:
            raw_id = data.get("id")
            if raw_id:
                inst._id = uuid.UUID(str(raw_id))
        except Exception:
            pass
        return inst

    # --------------------------------- misc -----------------------------------

    def __repr__(self) -> str:
        return f"<ProjectInformation '{self._title}' id={str(self._id)[:8]}>"
