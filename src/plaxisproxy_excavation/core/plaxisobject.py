from __future__ import annotations

from typing import (
    Any, Dict, Optional, Sequence, Set, Mapping, MutableSequence, MutableSet, Tuple,
    TypeVar, List, ClassVar, Type, Union, get_origin, get_args,
)
import uuid
import json
import inspect
from datetime import datetime, date
from enum import Enum

T = TypeVar("T", bound="SerializableBase")

# ############################## Meta #################################
class SerializableMeta(type):
    """
    Auto-discovers @property names to serialize if a subclass
    didn't explicitly provide __serialize_fields__.
    """
    def __new__(mcs, name: str, bases: Tuple[type, ...], dct: Dict[str, Any]) -> SerializableMeta:
        cls = super().__new__(mcs, name, bases, dct)
        if "__serialize_fields__" not in dct:
            discovered: List[str] = []
            # walk MRO but stop at object
            for c in inspect.getmro(cls):
                if c is object:
                    break
                for attr, member in c.__dict__.items():
                    if isinstance(member, property):
                        discovered.append(attr)
            # keep order, drop dups
            setattr(cls, "__serialize_fields__", list(dict.fromkeys(discovered)))
        return cls  # type: ignore[return-value]  # metaclass return type is the class itself


# ########################### Serializable ############################
class SerializableBase(metaclass=SerializableMeta):
    """
    Unifies (de)serialization for PLAXIS domain objects.
    - Serializes properties listed in __serialize_fields__ (auto-collected).
    - Falls back to __dict__ (excluding private "_*") if not present.
    - Handles nested SerializableBase / lists / tuples / dicts / uuid / dates / Enum.
    """
    __slots__: Tuple[str, ...] = tuple()  # subclasses may set their own
    _SERIAL_VERSION: int = 1
    __serialize_fields__: ClassVar[List[str]]

    # ########### Serialization ###########
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        fields = getattr(self, "__serialize_fields__", None)

        if fields:
            for k in fields:
                out[k] = self._serialize_value(getattr(self, k))
        else:
            for k, v in getattr(self, "__dict__", {}).items():
                if not k.startswith("_"):
                    out[k] = self._serialize_value(v)

        out["__type__"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        out["__version__"] = getattr(self, "_SERIAL_VERSION", 1)
        return out

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, SerializableBase):
            return value.to_dict()
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(i) for i in value]
        if isinstance(value, dict):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, Enum):
            return value.name
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        return value

    # ########### Deserialization ##########
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        # Remove meta keys that constructors don't expect
        data = {k: v for k, v in data.items() if not (isinstance(k, str) and k.startswith("__"))}

        # Use annotations when available, otherwise pass through
        type_hints: Dict[str, Any] = getattr(cls, "__annotations__", {}) or {}
        kwargs: Dict[str, Any] = {}
        for k, v in data.items():
            if k in type_hints:
                kwargs[k] = cls._deserialize_value(v, type_hints[k])
            else:
                # Keep tolerant behavior; if your __init__ doesn't accept extra keys,
                # change this to `continue`.
                kwargs[k] = v
        return cls(**kwargs)  # type: ignore[misc]

    @classmethod
    def _deserialize_value(cls, value: Any, expected_type: Any) -> Any:
        if value is None:
            return None

        origin = get_origin(expected_type)

        # Optional/Union[T, None] and general Union handling
        if origin is Union:
            args = [a for a in get_args(expected_type) if a is not type(None)]
            if not args:
                return None
            # Try the first non-None branch (for Optional this is enough)
            try:
                return cls._deserialize_value(value, args[0])
            except Exception:
                # Fallback: try each until one works
                for a in args[1:]:
                    try:
                        return cls._deserialize_value(value, a)
                    except Exception:
                        continue
                return value

        if origin is list:
            (item_t,) = get_args(expected_type) or (Any,)
            return [cls._deserialize_value(i, item_t) for i in value]

        if origin is tuple:
            args = get_args(expected_type)
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(cls._deserialize_value(i, args[0]) for i in value)
            return tuple(cls._deserialize_value(i, t) for i, t in zip(value, args))

        if origin is dict:
            k_t, v_t = get_args(expected_type) or (Any, Any)
            return {cls._deserialize_value(k, k_t): cls._deserialize_value(v, v_t) for k, v in value.items()}

        if isinstance(expected_type, type) and issubclass(expected_type, SerializableBase):
            return expected_type.from_dict(value)

        if expected_type is datetime:
            return datetime.fromisoformat(value)
        if expected_type is date:
            return date.fromisoformat(value)
        if expected_type is uuid.UUID:
            return uuid.UUID(value)
        if isinstance(expected_type, type) and issubclass(expected_type, Enum):
            return expected_type[value]

        try:
            return expected_type(value)  # type: ignore[call-arg]
        except Exception:
            return value  # be tolerant

    # ########### JSON helpers ############
    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls: Type[T], s: str) -> T:
        return cls.from_dict(json.loads(s))


# ######## Constants for copy scrubbing ########
_PRIMITIVES: Tuple[type, ...] = (str, bytes, int, float, bool, type(None))
_RUNTIME_FIELDS: Sequence[str] = ("plx_id", "_plaxis_ref", "_g_i", "_g_o", "_cache", "_log")
_ID_FIELDS: Sequence[str] = ("id", "_id", "uid", "_uid", "uuid", "_uuid", "serial", "_serial")


class PlaxisObject(SerializableBase):
    """
    Common base for all PLAXIS proxy objects.
    - GUID `id` for client-side tracking
    - `plx_id` to hold the server-side handle (set by mappers)
    - `name` and `comment`
    """
    __slots__ = ("_id", "_plx_id", "_name", "_comment")

    def __init__(self, name: str = "", comment: str = "") -> None:
        # if not isinstance(name, str) or not name:
        #     raise ValueError("name must be a non-empty string")
        self._id: str = str(uuid.uuid4())
        self._plx_id: Optional[object] = None  # runtime handle from PLAXIS
        self._name: str = name
        self._comment: str = comment

    # #################### copying with scrubbing ####################
    def copy(
        self: T,
        *,
        name: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        deep: bool = True,
        reset_runtime: bool = True,
        reset_ids: bool = True,
        exclude: Sequence[str] = (),
        memo: Optional[Dict[int, Any]] = None,
    ) -> T:
        """
        Create a copy with optional renaming and scrubbing runtime/id fields.
        Returns an instance of the same (sub)class.
        """
        import copy as _py_copy  # local import to avoid unused warning at module level
        new_obj = _py_copy.deepcopy(self, memo) if deep else _py_copy.copy(self)

        # rename
        if "name" not in exclude:
            old_name: Optional[str] = getattr(new_obj, "name", None)
            if name is not None:
                new_obj.name = name  # type: ignore[attr-defined]
            elif old_name is not None and (prefix or suffix):
                new_obj.name = f"{prefix}{old_name}{suffix}"  # type: ignore[attr-defined]

        # scrub runtime / ids recursively
        if reset_runtime or reset_ids:
            _plaxisobj_scrub(
                new_obj,
                reset_runtime=reset_runtime,
                reset_ids=reset_ids,
                exclude=set(exclude),
            )
        return new_obj

    # ########################### properties ###########################
    @property
    def id(self) -> str:
        """Client-side GUID."""
        return self._id
    
    @id.setter
    def id(self, id: str) -> None:
        """Set UUID Object."""
        self._id = id

    @property
    def plx_id(self) -> Optional[object]:
        """PLAXIS runtime handle (set by mappers)."""
        return self._plx_id

    @plx_id.setter
    def plx_id(self, value: Optional[object]) -> None:
        self._plx_id = value

    @property
    def name(self) -> str:
        """User-defined object name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        # avoid recursion: set the slot directly
        if not isinstance(value, str) or not value:
            raise ValueError("name must be a non-empty string")
        self._name = value

    @property
    def comment(self) -> str:
        """User note or description."""
        return self._comment

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', id='{self.id}')>"


def _plaxisobj_scrub(
    obj: object,
    *,
    reset_runtime: bool,
    reset_ids: bool,
    exclude: Set[str],
    _seen: Optional[Set[int]] = None,
) -> None:
    """
    Recursively scrub runtime fields and/or ID fields on obj.

    - Traverses mappings / sequences / user objects (via __dict__/__slots__).
    - Tolerant to read-only attributes.
    """
    if _seen is None:
        _seen = set()

    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)

    # Leaves
    if isinstance(obj, _PRIMITIVES):
        return

    # Mappings
    if isinstance(obj, Mapping):
        for _, v in obj.items():
            _plaxisobj_scrub(v, reset_runtime=reset_runtime, reset_ids=reset_ids, exclude=exclude, _seen=_seen)
        return

    # Sequences / sets
    if isinstance(obj, (list, tuple, set, frozenset, MutableSequence, MutableSet)):
        for v in obj:
            _plaxisobj_scrub(v, reset_runtime=reset_runtime, reset_ids=reset_ids, exclude=exclude, _seen=_seen)
        return

    # User object: scrub then recurse
    if reset_runtime:
        for f in _RUNTIME_FIELDS:
            if f in exclude:
                continue
            if hasattr(obj, f):
                try:
                    setattr(obj, f, None)
                except Exception:
                    pass

    if reset_ids:
        for f in _ID_FIELDS:
            if f in exclude:
                continue
            if hasattr(obj, f):
                try:
                    setattr(obj, f, None)
                except Exception:
                    pass

    # __dict__
    dct = getattr(obj, "__dict__", None)
    if isinstance(dct, dict):
        for k, v in dct.items():
            if k in exclude:
                continue
            _plaxisobj_scrub(v, reset_runtime=reset_runtime, reset_ids=reset_ids, exclude=exclude, _seen=_seen)

    # __slots__
    slots = getattr(obj, "__slots__", None)
    if slots:
        slot_names = slots if isinstance(slots, (list, tuple)) else (slots,)
        for s in slot_names:
            if s in exclude:
                continue
            try:
                v = getattr(obj, s)
            except Exception:
                continue
            _plaxisobj_scrub(v, reset_runtime=reset_runtime, reset_ids=reset_ids, exclude=exclude, _seen=_seen)
