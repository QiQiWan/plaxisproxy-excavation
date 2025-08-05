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

    class Time(Enum):
        S = "s"
        MIN = "min"
        H = "h"


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
        internal_force_unit: Units.Force,
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
        self._internal_force_unit = internal_force_unit
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
        # Create a copy to avoid modifying the original dictionary
        constructor_args = data.copy()

        # Convert unit strings back to Enum members
        constructor_args['length_unit'] = Units.Length(constructor_args['length_unit'])
        constructor_args['internal_force_unit'] = Units.Force(constructor_args['internal_force_unit'])
        constructor_args['time_unit'] = Units.Time(constructor_args['time_unit'])

        # Store the id to be set after instantiation
        instance_id = constructor_args.pop('id', None)

        # Create the instance
        instance = cls(**constructor_args)

        # Set the id from the original data
        if instance_id:
            instance._id = uuid.UUID(instance_id)

        return instance

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
            "internal_force_unit": self._internal_force_unit.value,
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
    def internal_force_unit(self):
        return self._internal_force_unit

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