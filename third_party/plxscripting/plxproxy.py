"""
Purpose: provide objects which act as a remote proxy to their
    actual Plaxis equivalents. The user is then able to create and mutate these
    objects using Python without requiring knowledge of the underlying
    communication.

Subversion data:
    $Id: plxproxy.py 21333 2016-02-22 14:14:52Z tj $
    $URL: https://tools.plaxis.com/svn/sharelib/trunk/PlxObjectLayer/Server/plxscripting/plxproxy.py $

Copyright (c) Plaxis bv. All rights reserved.

Unless explicitly acquired and licensed from Licensor under another
license, the contents of this file are subject to the Plaxis Public
License ("PPL") Version 1.0, or subsequent versions as allowed by the PPL,
and You may not copy or use this file in either source code or executable
form, except in compliance with the terms and conditions of the PPL.

All software distributed under the PPL is provided strictly on an "AS
IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the PPL for specific
language governing rights and limitations under the PPL.
"""

import abc

from .plx_scripting_exceptions import PlxScriptingError
from .plxobjects import PlxObjectPropertyList, PlxStagedIPList
from .selection import Selection
from .const import (COUNT, SUBLIST, INDEX, STAGED_PREFIX, SELECTION)


class PlxProxyObject_Abstract(object):
    """ Abstract base class for plaxis proxy objects. """

    def __init__(self, server, guid):
        self._attr_cache = {}
        self._server = server
        self._guid = guid

    @abc.abstractmethod
    def __repr__(self):
        """ Returns a representation of the object """
        return

    @property
    def attr_cache(self):
        self._ensure_cache_is_valid()
        return self._attr_cache

    @abc.abstractmethod
    def _ensure_cache_is_valid(self):
        """ Ensures that the attribute cache is up to date """
        return

    def get_cmd_line_repr(self):
        """
        Gets the command line representation of this object, taking into
        account the server's ability to convert GUIDs to their object names.
        """
        return self._guid

    def __dir__(self):
        """
        Return all attributes of the object (both from Plaxis and locally)
        """
        return dir(super(PlxProxyObject_Abstract, self)) + list(
            self.attr_cache)

    def get_equivalent(self, object=None):
        if object is None:
            object = self

        try:
            object_name = self._server.get_name_by_guid(object._guid)
        except KeyError:
            # Materials in output have different GUID's mostly because new materials are created to support
            # design approaches on different phases. We need to grab the real name of the object when getting the
            # equivalent object
            object_name = object.Name.value
        try:
            return self._server.get_named_object(object_name)
        except PlxScriptingError:
            pass
        raise AttributeError(
            "Requested object '{}' is not present".format(object_name))


class PlxProxyGlobalObject(PlxProxyObject_Abstract):
    """
    A single entity which represents all "global objects" in Plaxis. Resolution
    of the relevant global object target is done remotely by Plaxis.
    The user therefore does not require knowledge of the various global
    objects.
    """

    def __init__(self, server):
        super(PlxProxyGlobalObject, self).__init__(server, "")
        self._selection = Selection(server)

    def __repr__(self):
        return "<Global object>"

    def _getattr(self, attr_name):
        """
        Attempt to retrieve an attribute of one of the remote global objects.
        """
        if attr_name in self.attr_cache:
            return self.attr_cache[attr_name]

        # check if this is a named attribute (e.g. 'Points')
        try:
            # Note that in different modes different objects may have the same name,
            # and objects may be renamed.
            return self._server.get_named_object(attr_name)
        except PlxScriptingError:
            pass
        raise AttributeError("Requested attribute '{}' is not present".format(attr_name))

    def __dir__(self):
        self._ensure_cache_is_valid()
        return super().__dir__()

    @property
    def attr_cache(self):
        return self._attr_cache

    @property
    def selection(self):
        try:
            return self._getattr(SELECTION)
        except AttributeError:
            pass

        return self.__selection__

    @property
    def __selection__(self):
        # Selection may have been changed manually so refresh.
        self._selection.refresh()
        return self._selection

    @staticmethod
    def _is_model_group(proxy):
        if not isinstance(proxy, PlxProxyListable):
            return False

        # TODO: Not an ideal check, but this is needed to distinguish between
        # e.g. the Lines object and a Line_1 object. Both are listable, but
        # in the second case the points that make up the line shouldn't be
        # selected. Just the line.
        return proxy.TypeName.value.endswith('ModelGroup')

    @selection.setter
    def selection(self, value):
        # No check for the existence of another object called 'selection' is
        # done because assigning to a global object doesn't work anyway.
        self.__selection__ = value

    @__selection__.setter
    def __selection__(self, value):
        if value is None:
            self._selection.clear()
        elif PlxProxyGlobalObject._is_model_group(value):
            self._selection.set(list(value))
        elif isinstance(value, PlxProxyObject):
            self._selection.set([value])
        elif hasattr(value, '__iter__'):
            self._selection.set(value)
        elif isinstance(value, Selection):
            # Needed for +/- operators. Effectively a no-op
            self._selection = value
        else:
            raise PlxScriptingError("Can't set selection to '{}'".format(value))

    def __getattr__(self, attr_name):
        """
        Attempt to get it from cache, if it fails then re-fill cache and try one more time
        """
        try:
            return self._getattr(attr_name)
        except AttributeError:
            self._ensure_cache_is_valid()
            return self._getattr(attr_name)

    def _ensure_cache_is_valid(self):
        """
        Since the global object's attributes can change depending on the
        current mode (soil, structures, stages, etc.), it is necessary to
        refill the cache to ensure validity.
        """
        self._attr_cache = self._server.get_object_attributes(self)


class PlxProxyObject(PlxProxyObject_Abstract):
    """ A proxy Plaxis object """

    def __init__(self, server, guid, plx_type):
        super(PlxProxyObject, self).__init__(server, guid)

        self._plx_type = plx_type

    def __repr__(self):
        return "<{} {}>".format(self._plx_type, self._guid)

    def __getattr__(self, attr_name):
        """
        Returns the named attribute from Plaxis. These are either intrinsic
        property representations or method objects which can be called with
        arguments. If an attribute is not present, an exception is raised.
        """
        if attr_name in self.attr_cache:
            return self.attr_cache[attr_name]

        # Maybe it's a feature to be returned by its type name (e.g. myline.Beam).
        # These cannot be cached, because features can be added and removed at
        # runtime, unlike IPs.
        try:
            # Access UserFeatures through _attr_cache to prevent infinite recursion of __getattr__
            userfeatures = self._attr_cache['UserFeatures'].value
            for uf in userfeatures:
                featureTypeName = uf._plx_type
                if featureTypeName.startswith(STAGED_PREFIX):
                    featureTypeName = featureTypeName[len(STAGED_PREFIX):]

                if featureTypeName == attr_name:
                    return uf
        except Exception:  # don't catch BaseException and direct children (e.g. KeyboardInterrupt/SystemExit)
            pass

        # The requested attribute is not an attribute from the HTTP API (nor
        # another attribute of the python object).
        raise AttributeError(
            "Requested attribute '{}' is not present".format(attr_name))

    def __setattr__(self, name, value):
        """
        Sets the named attribute on the proxy object. This may be a proxy
        attribute or an attribute created within Python.

        There is a special case for those attributes which are required
        for the local objects. These must have an underscore prefix.
        Attributes which are set on the object without an underscore
        prefix will result in the cache being loaded, which is
        expensive.

        There is therefore an assumption that there will never be
        property names defined in Plaxis that have an underscore
        prefix. (Otherwise these could be "masked" here.)
        """
        # Check if the attribute is present
        attr_to_set = None
        try:
            attr_to_set = super(PlxProxyObject, self).__getattribute__(name)
        except AttributeError:
            pass

        # Either a normal Python attribute or a local protected attribute
        if attr_to_set is not None or name.startswith("_"):
            return super(PlxProxyObject, self).__setattr__(name, value)

        # Possibly a proxy attribute, so load the cache now
        if name in self.attr_cache:
            attr_to_set = self.attr_cache[name]
            # Allow setting of non-descriptor (i.e. a method attribute) to fail
            return attr_to_set.__set__(self, value)

        # Also allow setting of new attributes (i.e. those defined by user)
        return super(PlxProxyObject, self).__setattr__(name, value)

    def _ensure_cache_is_valid(self):
        # Regular proxy objects retain the same attributes throughout their
        # lifetime
        if not self._attr_cache:
            self._attr_cache = self._server.get_object_attributes(self)


class PlxProxyListable(object):
    """
    A mixin class which enables a proxy object to be listable.

    The mixin is totally dependent on being correctly mixed with a proxy
    object as it assumes that the server attribute is ready to use.
    """

    ITER_CACHE_COUNT = 100

    # It is important for the mixin to be able to accept the same arguments
    # as the class into which it is being "mixed". For example, if the
    # companion class is a regular PlxProxyObject, then the caller of the
    # mixin constructor should be able to assume that the proxy object receives
    # the relevant parameters.
    def __init__(self, *args, **kwargs):
        super(PlxProxyListable, self).__init__(*args, **kwargs)

    def __len__(self):
        return self._server.call_listable_method(self, COUNT)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # The server can process the original slice indices on its own
            return self._server.call_listable_method(
                self, SUBLIST, startindex=key.start, stopindex=key.stop)

        if not isinstance(key, int):
            raise TypeError("list indices must be integers, not {}".format(
                key.__class__.__name__))

        if key >= len(self):
            raise IndexError("list index out of range")

        return self._server.call_listable_method(self, INDEX, startindex=key)

    def __setitem__(self, key, value):
        return self._server.set_object_property(self[key], [value])

    def __iter__(self):
        # To prevent making a request for each element in an array, a slice is
        # requested instead. cache_start is used to indicate the starting index
        # of the next slice and gets incremented by ITER_CACHE_COUNT for each slice
        # that is requested. The iter_cache is exhausted by simply popping off
        # the front element, ensuring FIFO behavior.
        cache_start = 0
        length = len(self)
        while cache_start < length:
            cache_end = min(cache_start + PlxProxyListable.ITER_CACHE_COUNT, length)
            iter_cache = self._server.call_listable_method(self, SUBLIST,
                                                           startindex=cache_start,
                                                           stopindex=cache_end)
            cache_start = cache_end
            for element in iter_cache:
                yield element

    def _ensure_cache_is_valid(self):
        # Subobject properties are guaranteed not to be
        # removed / added other lifetime of the listable object
        super(PlxProxyListable, self)._ensure_cache_is_valid()

        if len(self) > 0:
            first_object = self[0]

            # We wish to query its attributes so it is pointless to continue if the
            # contained object is not a PlxProxyObject
            if not isinstance(first_object, PlxProxyObject_Abstract):
                return

            for attr_name, attr in self._server.get_object_attributes(first_object).items():
                if attr_name in self._attr_cache:
                    continue

                # Only support properties for now
                if not isinstance(attr, PlxProxyObjectProperty):
                    continue

                if isinstance(attr, PlxProxyIPStaged):
                    self._attr_cache[attr_name] = PlxStagedIPList(self._server, self, attr_name)
                else:
                    self._attr_cache[attr_name] = PlxObjectPropertyList(self._server, self, attr_name)


class PlxProxyValues(PlxProxyListable):
    """
    Same as PlxProxyListable but with context management so the TPlxValues
    object gets deleted.
    """

    def __enter__(self):
        return self

    def __exit__(self, type_, value, trace):
        self.delete()


class PlxProxyObjectMethod(object):
    """ A proxy method for a PlxProxyObject """

    def __init__(self, server, proxy_object, method_name):
        """ Create a proxy object method. """
        self._server = server

        self._proxy_object = proxy_object
        self._method_name = method_name

    def __call__(self, *params):
        """
        Method call on the target proxy object. The result of this call may
        be an exception, a list of new proxy objects, a boolean or a message.
        """
        return self._server.call_plx_object_method(
            self._proxy_object, self._method_name, params)


class PlxProxyMaterial(PlxProxyObject):
    """
    A proxy object that reinitializes its cache every time its used.

    This is necessary for soil materials because they can appear to add and
    remove intrinsic properties through the use of the strategy pattern.
    """

    def __init__(self, server, guid, plx_type):
        super(PlxProxyMaterial, self).__init__(server, guid, plx_type)
        server.add_proxy_to_reset(self)

    def reset_cache(self):
        self._attr_cache = None


class PlxProxyObjectProperty(PlxProxyObject):
    """ A proxy property for a PlxProxyObject """

    def __init__(self, server, guid, plx_type, property_name, owner):
        super(PlxProxyObjectProperty, self).__init__(server, guid, plx_type)

        self._owner = owner
        self._property_name = property_name
        # If the proxy property is a staged intr. prop., the primitive is stored in the _stagedIP_value.
        self._stagedIP_value = None

    def set_stagedIP_value(self, value):
        self._stagedIP_value = value

    @property
    def value(self):
        if self._stagedIP_value is not None:
            return self._stagedIP_value
        else:
            return self._get_value()

    def _get_value(self):
        return self._server.get_object_property(self._owner, self._property_name)

    def __str__(self):
        return str(self.value)

    def __bool__(self):
        return True if self.value else False

    __nonzero__ = __bool__  # compatibility with Py2.x

    def __set__(self, instance, value):
        return self._server.set_object_property(self, [value])


class PlxProxyIPBoolean(PlxProxyObjectProperty):
    """
    A proxy Boolean intrinsic property
    """

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other


class PlxProxyIPNumber(PlxProxyObjectProperty):
    """
    A proxy Number intrinsic property
    """

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __floordiv__(self, other):
        return self.value // other

    def __rfloordiv__(self, other):
        return other // self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __pow__(self, other):
        return pow(self.value, other)

    def __rpow__(self, other):
        return pow(other, self.value)

    def __mod__(self, other):
        return self.value % other

    def __rmod__(self, other):
        return other % self.value

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other


class PlxProxyIPInteger(PlxProxyIPNumber):
    """
    A proxy Integer intrinsic property
    """


class PlxProxyIPDouble(PlxProxyIPNumber):
    """
    A proxy Double intrinsic property
    """


class PlxProxyIPObject(PlxProxyObjectProperty):
    """
    A proxy intrinsic property object.

    Like other proxy intrinsic properties, there is a difference between the
    intrinsic property object and the value that that object "wraps".

    There is some additional complexity for this type of proxy intrinsic
    property because it makes the property attributes of the value
    available, as found on the standard POL command line. Its property
    attributes are merged with those of the value of the intrinsic property.

    In contrast, the method attributes of the value of the intrinsic property
    are not available from the intrinsic property without explicitly getting
    its value first.

    E.g.

    # Succeeds because this is a property attribute of the value
    print(my_pile.Parent.AxisFunction)

    # Fails, because this is a method attribute of the value
    my_pile.Parent.move(4, 4, 4)

    # Succeeds, because the method is being explicitly called on the value
    my_pile.Parent.value.move(4, 4, 4)
    """

    def __init__(self, *args, **kwargs):
        super(PlxProxyIPObject, self).__init__(*args, **kwargs)
        self._ip_attr_cache = {}

    def _get_value_properties_dict(self):
        value_props_dict = {}

        value = self.value  # cache so we don't do two roundtrips to the server
        if not isinstance(value, PlxProxyObject_Abstract):
            return value_props_dict  # No props for unassigned object

        for attr_name, attr in value.attr_cache.items():
            # Avoid adding attributes of the value that clash
            # with those of the intrinsic property, and the
            # methods.
            if not attr_name in dir(super(PlxProxyObject, self)):
                # Don't use hasattr on a PlxProxyIPObject referred by this one,
                # since calling hasattr will use __getattr__, potentially creating
                # an imense amount of recusion. This is especially true for the
                # PreviousPhase property of a phase.
                if isinstance(attr, PlxProxyIPObject):
                    value_props_dict[attr_name] = attr
                elif '__call__' not in attr.__dict__:
                    value_props_dict[attr_name] = attr

        return value_props_dict

    def _ensure_cache_is_valid(self):
        """
        Proxy intrinsic property objects merge their attributes with the
        property attributes of their values.

        Since the value of the intrinsic property can change, the
        cache of the value is required to be up to date.
        """
        super(PlxProxyIPObject, self)._ensure_cache_is_valid()

        if not self._ip_attr_cache:
            self._ip_attr_cache = self._attr_cache

        # Merge the value properties with the intrinsic property
        # every time, since the value may change any time.
        self._attr_cache = dict(self._ip_attr_cache,
                                **self._get_value_properties_dict())

    def __len__(self):
        value = self.value
        if not isinstance(value, PlxProxyListable):
            raise TypeError("object of type '{}' has no len()".format(
                type(value)))
        return len(value)

    def __getitem__(self, index):
        value = self.value
        if not isinstance(value, PlxProxyListable):
            raise TypeError("'{}' object is not subscriptable".format(
                type(value)))
        return value[index]

    def __iter__(self):
        value = self.value
        if not isinstance(value, PlxProxyListable):
            raise TypeError("'{}' object is not iterable".format(
                type(value)))
        return iter(value)


class PlxProxyIPEnumeration(PlxProxyObjectProperty):
    """
    A proxy Enumeration intrinsic property
    """

    # TODO: Make comparison method(s) compare
    # enumerations within one type of enumeration
    # i.e. pile connection enumeration 1 should not
    # equal some other enumeration 1!
    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def enum_dict(self):
        type_dict = type(self).__dict__
        return {k: type_dict[k] for k in type_dict
                if not k.startswith('__')}

    @property
    def strvalue(self):
        value = self.value
        for k, v in self.enum_dict.items():
            if value == v:
                return k

        # This exception indicates a problem with how the enum keys are retrieved.
        # It will not be a user error.
        raise ValueError("Invalid enum value '{}'".format(value))

    @strvalue.setter
    def strvalue(self, string):
        enum_dict = self.enum_dict
        if string not in enum_dict:
            key_names = ', '.join(self.enum_dict.keys())
            raise ValueError("Invalid enum name '{}', valid values are '{}'".format(
                string, key_names))

        # Access type descriptor setter directly since assigning to self
        # doesn't work.
        self.__set__(None, enum_dict[string])


class PlxProxyIPText(PlxProxyObjectProperty):
    """
    A proxy Text intrinsic property
    """

    def __add__(self, other):
        return self.value + other

    def __getitem__(self, index):
        return self.value[index]

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other


class PlxProxyIPStaged(PlxProxyObjectProperty):
    """
    A proxy staged intrinsic property
    """

    def __getitem__(self, phase_object):
        # by defining __getitem__, the object can be iterated in a for loop with an integer key starting from 0.
        # if no exception is raised, the for loop is infinite. The following check solves this.
        if not isinstance(phase_object, PlxProxyObject):
            raise TypeError('Expected phase object key')

        return self._server.get_object_property(
            self._owner, self._property_name, phase_object)

    def __setitem__(self, phase_object, value):
        if value is None:  # cannot set staged to None, as the Plaxis command line doesn't support the concept of None
            return None

        # ProxyObjects that are not ProxyIP need to be de-referenced by its value
        if isinstance(value, PlxProxyObjectProperty) and not isinstance(value, PlxProxyIPStaged):
            value = value.value

        return self._server.set_object_property(
            self, [phase_object, value])
