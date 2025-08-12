# -*- coding: utf-8 -*-
"""plx.projectinformation - metadata block for a Plaxis modelling job

This revision modernises the original *ProjectInformation* class while keeping
its public read-only property interface backward-compatible.

*   Adds basic **input validation** (non-empty title / file path, numeric ranges
    for bounding box, positive γw).  Bad input raises *ValueError* early.
*   Introduces *unit* enums so that downstream code can avoid free-text unit
    typos (``Units.Length.M`` etc.).
*   Provides lightweight :py:meth:`to_dict` for JSON / YAML serialisation and
    :py:meth:`summary` for quick logging.
*   Keeps a stable UUID and minimal memory footprint by using **__slots__**.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Dict, Any, Type
from ..core.plaxisobject import SerializableBase

__all__ = ["Units", "ProjectInformation"]


class Units:
    """Namespace for unit enums grouped by physical dimension."""

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


class ProjectInformation(SerializableBase):
    """Read-only container for model metadata."""

    __slots__ = (
        "_id", "_title", "_company", "_dir", "_file_name", "_comment",
        "_model", "_element", "_length_unit", "_internal_force_unit",
        "_time_unit", "_gamma_water", "_x_min", "_x_max", "_y_min", "_y_max",
    )

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------
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
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:

        # basic value checks -------------------------------------------------
        if not title:
            raise ValueError("Project title must not be empty")
        if not dir:
            raise ValueError("Project directory must not be empty")
        if not file_name:
            raise ValueError("File name must not be empty")
        if gamma_water <= 0:
            raise ValueError("gamma_water must be positive (kN/m³)")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("(x_max, y_max) must exceed (x_min, y_min)")

        self._id = uuid.uuid4()
        self._title = title
        self._company = company
        self._dir = dir
        self._file_name = file_name
        self._comment = comment
        self._model = model
        self._element = element
        self._length_unit = length_unit
        self._force_unit = force_unit
        self._stress_unit = stress_unit
        self._time_unit = time_unit
        self._gamma_water = float(gamma_water)
        self._x_min = float(x_min)
        self._x_max = float(x_max)
        self._y_min = float(y_min)
        self._y_max = float(y_max)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Creates a ProjectInformation instance from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing the project information data.

        Returns:
            ProjectInformation: An instance of the ProjectInformation class.
        """
        units_block = data.get("units", {}) or {}

        def parse_enum(val, enum_cls, default):
            if val is None:
                return default
            if isinstance(val, enum_cls):
                return val
            if isinstance(val, str):
                s = val.strip()
                if hasattr(enum_cls, s):
                    try:
                        return getattr(enum_cls, s)
                    except Exception:
                        pass
                try:
                    return enum_cls(s)
                except Exception:
                    pass
            try:
                return enum_cls(val)
            except Exception:
                return default

        length_unit = parse_enum(
            data.get("length_unit", units_block.get("length")),
            Units.Length, Units.Length.M
        )
        force_unit = parse_enum(
            data.get("force_unit", units_block.get("force")),
            Units.Force, Units.Force.KN
        )
        stress_unit = parse_enum(
            data.get("stress_unit", units_block.get("stress")),
            Units.Stress, Units.Stress.KPA
        )
        time_unit = parse_enum(
            data.get("time_unit", units_block.get("time")),
            Units.Time, Units.Time.DAY
        )

        # ---- 关键修复：dir 的容错读取 + 非空默认值 ----
        dir_val = (
            data.get("dir")
            or data.get("project_dir")
            or data.get("directory")
            or "."
        )

        # 其余必填字段给出稳妥默认（按你的 __init__ 需要增减）
        str_default = "C://example_project"
        gamma_water_default = 10.0
        x_min_default = 0.0
        y_min_default = 0.0
        x_max_default = 80.0
        y_max_default = 80.0

        constructor_args: Dict[str, Any] = {
            "title":       data.get("title", str_default),
            "company":     data.get("company", str_default),
            "dir":         dir_val,                 # <- 不再为空
            "file_name":   data.get("file_name", str_default),
            "comment":     data.get("comment", str_default),
            "model":       data.get("model", str_default),
            "element":     data.get("element", str_default),
            "gamma_water": float(data.get("gamma_water", gamma_water_default)),
            "x_min":       float(data.get("x_min", x_min_default)),
            "x_max":       float(data.get("x_max", x_max_default)),
            "y_min":       float(data.get("y_min", y_min_default)),
            "y_max":       float(data.get("y_max", y_max_default)),
            "length_unit": length_unit,
            "force_unit":  force_unit,
            "stress_unit": stress_unit,
            "time_unit":   time_unit,
        }

        return cls(**constructor_args)
    
    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def bounding_box(self):
        """Return domain bounding box as ``(x_min, x_max, y_min, y_max)``."""
        return self._x_min, self._x_max, self._y_min, self._y_max
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain Python dict (for JSON / YAML export)."""
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
            "stress_unit": self._stress_unit,
            "force_unit": self._force_unit.value,
            "time_unit": self._time_unit.value,
            "gamma_water": self._gamma_water,
            "x_min": self._x_min,
            "x_max": self._x_max,
            "y_min": self._y_min,
            "y_max": self._y_max,
        }

    def summary(self) -> str:
        return (
            f"{self._title} | {self._company} | γw = {self._gamma_water:g} kN/m³ | "
            f"domain: [{self._x_min}, {self._x_max}] × [{self._y_min}, {self._y_max}] {self._length_unit.value}"
        )

    # ------------------------------------------------------------------
    # Representation / read-only properties
    # ------------------------------------------------------------------
    def __repr__(self):
        return f"<ProjectInformation '{self._title}' id={str(self._id)[:8]}>"

    @property
    def id(self):
        return self._id

    @property
    def title(self):
        return self._title

    @property
    def company(self):
        return self._company

    @property
    def dir(self):
        return self._dir

    @property
    def file_name(self):
        return self._file_name

    @property
    def comment(self):
        return self._comment

    @property
    def model(self):
        return self._model

    @property
    def element(self):
        return self._element

    @property
    def length_unit(self):
        return self._length_unit

    @property
    def force_unit(self):
        return self._force_unit

    @property
    def stress_unit(self):
        return self._stress_unit

    @property
    def time_unit(self):
        return self._time_unit

    @property
    def gamma_water(self):
        return self._gamma_water

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    @property
    def y_min(self):
        return self._y_min

    @property
    def y_max(self):
        return self._y_max