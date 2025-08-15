import json
import uuid
from datetime import datetime, date
from typing import Any, get_origin, get_args, Dict,TypeVar, Type, ClassVar, List, Optional, Sequence, Set, Mapping, MutableSequence, MutableSet
import copy as _py_copy
from enum import Enum
import inspect

# A TypeVar used for type hinting, indicating that from_dict will return
# an instance of the class it is called on.
T = TypeVar('T', bound='SerializableBase')

class SerializableMeta(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        # Check if the user has manually defined __serialize_fields__ in the current class
        if '__serialize_fields__' not in dct:
            # If not, we perform auto-discovery
            discovered_fields = []
            for c in inspect.getmro(cls):
                if c is object:
                    continue
                for attr_name, member in c.__dict__.items():
                    if isinstance(member, property):
                        discovered_fields.append(attr_name)

            unique_fields = list(dict.fromkeys(discovered_fields))
            
            # Use setattr() for explicit dynamic attribute assignment.
            # This is idiomatic for metaprogramming and understood by static analyzers.
            setattr(cls, '__serialize_fields__', unique_fields)
            
        return cls
    
class SerializableBase(metaclass=SerializableMeta):
    """
    A base class that provides unified serialization and deserialization functionality.

    Subclasses can easily convert between dictionary/JSON representations and
    object instances. It automatically handles primitive types, datetime, date,
    UUID, lists, and nested SerializableBase objects.
    """
    _SERIAL_VERSION: int = 1
    __serialize_fields__: ClassVar[List[str]]

    # ---------------------------------------------------------------------------
    # Unified Serialization Helpers
    # ---------------------------------------------------------------------------
    # We define these as static methods because their operations do not depend on
    # any specific object instance (i.e., they don't need `self`).
    # Placing them inside the class organizes the code logically, showing they
    # serve the class's serialization logic.
    # The leading underscore _ indicates that they are internal helper methods,
    # not intended for direct external use.

    @staticmethod
    def _uuid_to_str(u: Optional[uuid.UUID]) -> Optional[str]:
        """Converts a UUID object to a string, handling None values."""
        return str(u) if u is not None else None

    @staticmethod
    def _str_to_uuid(s: Optional[str]) -> Optional[uuid.UUID]:
        """Converts a string to a UUID object, handling None values."""
        return uuid.UUID(s) if s is not None and s != '' else None

    # ---------------------------------------------------------------------------
    # Core Serialization Methods
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the current object instance into a dictionary.

        New Logic:
        1. It first looks for a `__serialize_fields__` attribute. If it exists, it uses
        this as the list of fields to serialize. This allows fields exposed
        via @property to be serialized correctly.
        2. If `__serialize_fields__` does not exist, it falls back to the old logic of
        iterating through __dict__ and ignoring underscore-prefixed attributes.
        """
        obj_dict = {}

        # Check for the declarative list of fields
        if hasattr(self, '__serialize_fields__'):
            field_names = self.__serialize_fields__
            for key in field_names:
                # getattr(self, key) will automatically call the @property's getter method
                value = getattr(self, key)
                obj_dict[key] = self._serialize_value(value)
        else:
            # Fallback to the old convention-based logic using __dict__
            for key, value in self.__dict__.items():
                if key.startswith('_'):
                    continue
                obj_dict[key] = self._serialize_value(value)

        obj_dict["__type__"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        obj_dict["__version__"] = getattr(self, "_SERIAL_VERSION", 1)

        return obj_dict

    def _serialize_value(self, value: Any) -> Any:
        """Recursively serializes a single value."""
        if isinstance(value, SerializableBase):
            return value.to_dict()
        # MODIFIED: Combined list and tuple handling
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        # NEW: Added dictionary handling
        elif isinstance(value, dict):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, uuid.UUID):
            return self._uuid_to_str(value)
        elif isinstance(value, Enum):
            return value.name
        elif hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        else:
            return value

    # ---------------------------------------------------------------------------
    # Core Deserialization Methods
    # ---------------------------------------------------------------------------

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """
        Creates and returns a class instance (cls) from a dictionary.
        It uses the class's type annotations (__annotations__) to intelligently
        convert data types.
        """
        # Get the class's type annotations, e.g., {'name': <class 'str'>, 'user_id': <class 'uuid.UUID'>}
        type_hints = cls.__annotations__

        constructor_args = {}
        for key, value in data.items():
            if key in type_hints:
                # Deserialize the value based on the type annotation
                expected_type = type_hints[key]
                constructor_args[key] = cls._deserialize_value(value, expected_type)
            else:
                # If there's no type annotation, use the original value directly
                constructor_args[key] = value

        return cls(**constructor_args)

    @classmethod
    def _deserialize_value(cls: Type[T], value: Any, expected_type: Type) -> Any:
        """Recursively deserializes a single value based on the expected type."""
        if value is None:
            return None

        origin_type = get_origin(expected_type)

        if origin_type is list:
            item_type = get_args(expected_type)[0]
            return [cls._deserialize_value(item, item_type) for item in value]
        
        # NEW: Added tuple handling
        if origin_type is tuple:
            # Handles Tuple[type, type, ...] and Tuple[type, ...]
            item_types = get_args(expected_type)
            if len(item_types) == 2 and item_types[1] is Ellipsis:  # Variadic tuple e.g. Tuple[int, ...]
                return tuple(cls._deserialize_value(item, item_types[0]) for item in value)
            # Fixed-length tuple e.g. Tuple[str, int]
            return tuple(cls._deserialize_value(item, item_types[i]) for i, item in enumerate(value))

        # NEW: Added dictionary handling
        if origin_type is dict:
            key_type, value_type = get_args(expected_type)
            return {
                cls._deserialize_value(k, key_type): cls._deserialize_value(v, value_type)
                for k, v in value.items()
            }

        if isinstance(expected_type, type) and issubclass(expected_type, SerializableBase):
            return expected_type.from_dict(value)

        if expected_type is datetime:
            return datetime.fromisoformat(value)

        if expected_type is date:
            return date.fromisoformat(value)

        if expected_type is uuid.UUID:
            return cls._str_to_uuid(value)
        
        if isinstance(expected_type, type) and issubclass(expected_type, Enum):
            return expected_type[value]

        return expected_type(value)

    # ---------------------------------------------------------------------------
    # Convenient JSON Helper Methods
    # ---------------------------------------------------------------------------

    def to_json(self, **kwargs) -> str:
        """Converts the object to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Creates an object instance from a JSON string."""
        return cls.from_dict(json.loads(json_str))
    

class PlaxisObject(SerializableBase):
    """
    The common base class of PLAXIS objects。

    This class encapsulates all the common core attributes shared by all PLAXIS proxy objects,
    including a unique internal ID, a handle (plx_id) for the PLAXIS server reference,
    and a user-defined name.

    Attributes:
        _id (str): A unique internal identifier (UUID) used to track the object within the session.
        _plx_id (Optional[str]): The object handle returned by the PLAXIS script interface.
        This value is None until the object is synchronized to the PLAXIS server.
        _name (str): The name assigned by the user to this object, which will also be used in PLAXIS.
    """
    def __init__(self, name: str, comment: str):
        """
        Initialize PlaxisObject。
        """
        if not isinstance(name, str) or not name:
            raise ValueError("The type of name must be a string!")
            
        self._id: str = str(uuid.uuid4())
        self._plx_id: Optional[str] = None
        self._name: str = name
        self._comment: str = comment

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
        Create a copy of this object with optional renaming and scrubbing of runtime/ID fields.

        Parameters
        ----------
        name : Optional[str]
            Explicit new name for the copy. If None, `prefix`/`suffix` will be applied to the
            original name when available.
        prefix : str
            Prefix to prepend to the original name (only used if `name` is None and
            the object has a `name` attribute).
        suffix : str
            Suffix to append to the original name (only used if `name` is None and
            the object has a `name` attribute).
        deep : bool
            If True (default), perform a deep copy. If False, perform a shallow copy.
        reset_runtime : bool
            If True (default), scrub runtime-only fields (e.g., `plx_id`, internal handles).
        reset_ids : bool
            If True (default), scrub persistent identifiers (e.g., `id`, `uid`, `uuid`).
        exclude : Sequence[str]
            Field names to exclude from renaming/scrubbing (e.g., ["name"]).
        memo : Optional[Dict[int, Any]]
            Optional memo dictionary forwarded to `copy.deepcopy`.

        Returns
        -------
        T
            A new instance of the same (sub)class with requested adjustments applied.
        """
        new_obj = _py_copy.deepcopy(self, memo) if deep else _py_copy.copy(self)

        # (1) Apply renaming if requested
        if "name" not in exclude:
            try:
                old_name = getattr(new_obj, "name", None)
                if name is not None:
                    setattr(new_obj, "name", name)
                elif old_name is not None and (prefix or suffix):
                    setattr(new_obj, "name", f"{prefix}{old_name}{suffix}")
            except Exception:
                # Be tolerant: some subclasses may not have a writable `name`
                pass

        # (2) Scrub runtime-only fields and/or persistent IDs recursively
        if reset_runtime or reset_ids:
            _plaxisobj_scrub(
                new_obj,
                reset_runtime=reset_runtime,
                reset_ids=reset_ids,
                exclude=set(exclude),
            )

        return new_obj

    @property
    def id(self) -> str:
        """Get the unique ID of the object."""
        return self._id

    @property
    def plx_id(self) -> Optional[str]:
        """Get the handle of Plaxis object."""
        return self._plx_id
    
    @plx_id.setter
    def plx_id(self, value):
        self._plx_id = value
 
    @property
    def name(self) -> str:
        """Get the name of Plaxis object."""
        return self._name
    
    @name.setter
    def name(self, value):
        self.name = value

    @property
    def comment(self) -> str:
        """Get comment or description."""
        return self._comment

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', id='{self.id}')>"


# -------- Internal helpers (keep in the same module) ---------------------------------

# Runtime-only fields that should never survive a copy used for new Plaxis mappings
_RUNTIME_FIELDS: Sequence[str] = (
    "plx_id",          # Plaxis object reference (assigned at runtime by mappers)
    "_plaxis_ref",     # possible internal Plaxis reference alias
    "_g_i", "_g_o",    # Plaxis input/output handles (if present)
    "_cache", "_log",  # any caches or loggers attached on the instance
)

# Stable identifier fields that typically should not be duplicated for a new instance
_ID_FIELDS: Sequence[str] = (
    "id", "_id", "uid", "_uid", "uuid", "_uuid",
    "serial", "_serial",
)

# Treat these as leaves for the scrub walk
_PRIMITIVES = (str, bytes, int, float, bool, type(None))


def _plaxisobj_scrub(
    obj: Any,
    *,
    reset_runtime: bool,
    reset_ids: bool,
    exclude: Set[str],
    _seen: Optional[Set[int]] = None,
) -> None:
    """
    Recursively scrub runtime fields and/or ID fields on `obj`.

    The function traverses:
      - mappings (dict-like)
      - sequences/sets
      - user objects (via `__dict__` and `__slots__`)

    Notes
    -----
    - This function assumes `obj` has already been (deep)copied when needed.
    - It is careful to avoid infinite recursion via an `_seen` set.
    """
    if _seen is None:
        _seen = set()

    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)

    # Primitive leaves: nothing to scrub below
    if isinstance(obj, _PRIMITIVES):
        return

    # Mappings: walk values
    if isinstance(obj, Mapping):
        for _, v in obj.items():
            _plaxisobj_scrub(v, reset_runtime=reset_runtime, reset_ids=reset_ids, exclude=exclude, _seen=_seen)
        return

    # Sequences/sets: walk elements
    if isinstance(obj, (list, tuple, set, frozenset, MutableSequence, MutableSet)):
        for v in obj:
            _plaxisobj_scrub(v, reset_runtime=reset_runtime, reset_ids=reset_ids, exclude=exclude, _seen=_seen)
        return

    # For user objects: scrub fields, then recurse into attributes
    if reset_runtime:
        for f in _RUNTIME_FIELDS:
            if f in exclude:
                continue
            if hasattr(obj, f):
                try:
                    setattr(obj, f, None)
                except Exception:
                    # Some slots may be read-only; ignore quietly
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

    # Recurse attributes exposed via __dict__
    dct = getattr(obj, "__dict__", None)
    if isinstance(dct, dict):
        for k, v in dct.items():
            if k in exclude:
                continue
            _plaxisobj_scrub(v, reset_runtime=reset_runtime, reset_ids=reset_ids, exclude=exclude, _seen=_seen)

    # Recurse attributes exposed via __slots__
    slots = getattr(obj, "__slots__", None)
    if slots:
        # Normalize to iterable
        slot_names = slots if isinstance(slots, (list, tuple)) else (slots,)
        for s in slot_names:
            if s in exclude:
                continue
            try:
                v = getattr(obj, s)
            except Exception:
                continue
            _plaxisobj_scrub(v, reset_runtime=reset_runtime, reset_ids=reset_ids, exclude=exclude, _seen=_seen)