from __future__ import annotations
from typing import Dict, Any, Type, TypeVar, Tuple, Optional
from enum import Enum
import uuid

__all__ = ["Units", "ProjectInformation"]

T = TypeVar("T", bound="ProjectInformation")


class Units:
    """
    Unit groups used by the project. Keep it small and pragmatic; extend as needed.
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
    Lightweight project metadata container with sensible defaults.

    Goals:
      - Minimal required input: only `title`.
      - Optional parameters default to a practical SI setup (m, kN, kPa, day).
      - Size-first API: pass `x_size` / `y_size` to set project extents quickly.
      - Backward-compatible serialization.

    Examples:
      # 1) Minimal (defaults to SI, 3D, 10-noded, box 50x80 from (0,0))
      proj = ProjectInformation(title="Basement Excavation")

      # 2) Explicit XY size from custom origin
      proj = ProjectInformation(title="Demo", x_min=-25, y_min=-40, x_size=50, y_size=80)

      # 3) Full control (units & bounds)
      proj = ProjectInformation(
          title="Demo",
          length_unit=Units.Length.M, force_unit=Units.Force.KN,
          stress_unit=Units.Stress.KPA, time_unit=Units.Time.DAY,
          x_min=0, y_min=0, x_max=120, y_max=80
      )
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
        *,
        # High-level info (optional)
        company: str = "Unknown",
        dir: str = ".",
        file_name: str = "project.p3d",
        comment: str = "",
        model: str = "3D",
        element: str = "10-noded",
        # Units (optional; SI defaults)
        length_unit: Units.Length = Units.Length.M,
        force_unit: Units.Force = Units.Force.KN,
        stress_unit: Units.Stress = Units.Stress.KPA,
        time_unit: Units.Time = Units.Time.DAY,
        gamma_water: float = 9.81,
        # Bounds: either (x_min/x_max & y_min/y_max) OR (x_min/y_min + x_size/y_size)
        x_min: float = 0.0,
        y_min: float = 0.0,
        x_max: Optional[float] = None,
        y_max: Optional[float] = None,
        x_size: Optional[float] = 50.0,
        y_size: Optional[float] = 80.0,
    ) -> None:
        # Required
        if not title:
            raise ValueError("Project title must not be empty.")
        if gamma_water <= 0:
            raise ValueError("gamma_water must be positive.")

        # Resolve bounds: prefer explicit max; otherwise compute from size
        if x_max is None:
            if x_size is None:
                x_size = 50.0
            x_max = float(x_min) + float(x_size)
        if y_max is None:
            if y_size is None:
                y_size = 80.0
            y_max = float(y_min) + float(y_size)

        if x_max <= x_min or y_max <= y_min:
            raise ValueError("(x_max, y_max) must be greater than (x_min, y_min).")

        # Store
        self._id = uuid.uuid4()
        self._title, self._company = title, company
        self._dir, self._file_name, self._comment = dir, file_name, comment
        self._model, self._element = model, element
        self._length_unit, self._force_unit = length_unit, force_unit
        self._stress_unit, self._time_unit = stress_unit, time_unit
        self._gamma_water = float(gamma_water)
        self._x_min, self._x_max = float(x_min), float(x_max)
        self._y_min, self._y_max = float(y_min), float(y_max)

        # Set by the Mapper after PLAXIS creation (group/handle/object)
        self.plx_id = None

    # --------------------------- read-only properties ---------------------------

    @property
    def id(self) -> uuid.UUID:
        return self._id

    @property
    def title(self) -> str:
        return self._title

    @property
    def company(self) -> str:
        return self._company

    @property
    def dir(self) -> str:
        return self._dir

    @property
    def file_name(self) -> str:
        return self._file_name

    @property
    def comment(self) -> str:
        return self._comment

    @property
    def model(self) -> str:
        return self._model

    @property
    def element(self) -> str:
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
        """Return (x_min, x_max, y_min, y_max)."""
        return self._x_min, self._x_max, self._y_min, self._y_max

    @property
    def x_size(self) -> float:
        return self._x_max - self._x_min

    @property
    def y_size(self) -> float:
        return self._y_max - self._y_min

    @classmethod
    def default_si(cls, title: str, **kwargs) -> "ProjectInformation":
        """Convenience constructor for a typical SI, 3D setup with 50x80 box."""
        return cls(title=title, **kwargs)

    # ------------------------------- serialization ----------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict; enums are stored via `.value`."""
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
        Re-create from dict. Supports legacy keys and tolerates unit strings or enum names.
        """

        def _parse_enum(val: Any, enum_cls: Any, default: Any) -> Any:
            if val is None:
                return default
            if isinstance(val, enum_cls):
                return val
            if isinstance(val, str):
                s = val.strip()
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

        # Directory aliases for backward compatibility
        dir_val = data.get("dir") or data.get("project_dir") or data.get("directory") or "."

        length_unit = _parse_enum(data.get("length_unit"), Units.Length, Units.Length.M)
        force_unit  = _parse_enum(data.get("force_unit"),  Units.Force,  Units.Force.KN)
        stress_unit = _parse_enum(data.get("stress_unit"), Units.Stress, Units.Stress.KPA)
        time_unit   = _parse_enum(data.get("time_unit"),   Units.Time,   Units.Time.DAY)

        # Prefer explicit bounds; otherwise use sizes if present; otherwise defaults
        x_min = float(data.get("x_min", 0.0))
        y_min = float(data.get("y_min", 0.0))
        x_max = data.get("x_max")
        y_max = data.get("y_max")
        x_size = data.get("x_size")
        y_size = data.get("y_size")

        kwargs: Dict[str, Any] = dict(
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
            x_min=x_min, y_min=y_min,
        )

        # Carry either max (preferred if present) or size
        if x_max is not None:
            kwargs["x_max"] = float(x_max)
        else:
            kwargs["x_size"] = float(x_size) if x_size is not None else 50.0

        if y_max is not None:
            kwargs["y_max"] = float(y_max)
        else:
            kwargs["y_size"] = float(y_size) if y_size is not None else 80.0

        inst = cls(**kwargs)

        # Preserve incoming ID if provided
        try:
            raw_id = data.get("id")
            if raw_id:
                inst._id = uuid.UUID(str(raw_id))
        except Exception:
            pass
        return inst

    # --------------------------------- misc -----------------------------------

    def __repr__(self) -> str:
        return f"<ProjectInformation '{self._title}' {self._model} {self._element} {self.x_size:.0f}x{self.y_size:.0f} id={str(self._id)[:8]}>"
