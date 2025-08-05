"""
Purpose: provide objects that are not proxies of Plaxis objects but wrappers
    to manipulate Plaxis objects in a Pythonic way.

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

from .const import (MEMBERSUBLIST, MEMBERINDEX)
# used to import PlxProxyObject, but not referencing it directly,
# otherwise cannot import because of circular reference
from . import plxproxy


class PlxObjectPropertyList:
    """
    A container-type object that iterates over the property x of each object
    contained in a listable owner (i.e. [List[0].x ... List[-1].x])

    When its value is queried, it can return the values of all properties
    in a single request in an actual Python list
    """

    ITER_CACHE_COUNT = 100

    def __init__(self, server, listable_parent, property_name, phase_object=None):
        self._server = server
        self._listable_parent = listable_parent
        self._owner_objects_cache = [None] * len(listable_parent)
        self._property_name = property_name
        self._phase_object = phase_object
        self._property_type = None

    def __repr__(self):
        if self._property_type is None:
            try:
                self._ensure_objects_cache_exists_for_key(0)
                self._property_type = self._owner_objects_cache[0].__getattr__(self._property_name)._plx_type
            except Exception:
                self._property_type = 'Property'

        return "<{} list>".format(self._property_type)

    def __len__(self):
        return len(self._listable_parent)

    def _ensure_objects_cache_exists_for_key(self, key):
        if isinstance(key, slice):
            for in_cache in self._owner_objects_cache[key]:
                if in_cache is None:
                    self._owner_objects_cache[key] = self._listable_parent[key]
                    break

        elif isinstance(key, int):
            if key >= len(self):
                raise IndexError("list index out of range")

        else:
            raise TypeError("list indices must be integers, not {}".format(
                key.__class__.__name__))

        self._owner_objects_cache[key] = self._listable_parent[key]

    def _get_staged_properties_for_key(self, key):
        if self._phase_object is None:
            raise TypeError("Cannot query staged properties without a phase object")

        self._ensure_objects_cache_exists_for_key(key)

        if isinstance(key, slice):
            return self._server.get_objects_property(proxy_objects=self._owner_objects_cache[key],
                                                     prop_name=self._property_name,
                                                     phase_object=self._phase_object)

        if not isinstance(key, int):
            raise TypeError("list indices must be integers, not {}".format(
                key.__class__.__name__))

        if key >= len(self):
            raise IndexError("list index out of range")

        return self._server.get_object_property(proxy_object=self._owner_objects_cache[key],
                                                prop_name=self._property_name,
                                                phase_object=self._phase_object)

    def _get_properties_for_key(self, key):
        self._ensure_objects_cache_exists_for_key(key)

        if isinstance(key, slice):
            return self._server.call_listable_method(
                self._listable_parent, MEMBERSUBLIST, startindex=key.start, stopindex=key.stop,
                property_name=self._property_name)

        if not isinstance(key, int):
            raise TypeError("list indices must be integers, not {}".format(
                key.__class__.__name__))

        if key >= len(self):
            raise IndexError("list index out of range")

        return self._server.call_listable_method(self._listable_parent, MEMBERINDEX, startindex=key,
                                                 property_name=self._property_name)

    def __getitem__(self, key):
        if self._phase_object is not None:
            return self._get_staged_properties_for_key(key)
        else:
            return self._get_properties_for_key(key)

    def __setitem__(self, key, value):
        if self._phase_object is not None:
            params = [self._phase_object, value]
        else:
            params = [value]

        property = self._get_properties_for_key(key)
        return self._server.set_object_property(property, params)

    def __iter__(self):
        # To prevent making a request for each element in an array, a slice is
        # requested instead. cache_start is used to indicate the starting index
        # of the next slice and gets incremented by ITER_CACHE_COUNT for each slice
        # that is requested. The iter_cache is exhausted by simply popping off
        # the front element, ensuring FIFO behavior.
        cache_start = 0
        length = len(self)
        while cache_start < length:
            cache_end = min(cache_start + PlxObjectPropertyList.ITER_CACHE_COUNT, length)
            iter_cache = self[cache_start:cache_end]
            cache_start = cache_end
            for element in iter_cache:
                yield element

    @property
    def value(self):
        return self._get_value()

    def _get_value(self):
        plx_objects_in_list = self._listable_parent[0:len(self)]
        properties = self._server.get_objects_property(plx_objects_in_list, self._property_name, self._phase_object)

        # For staged IPs, the returned object is a staged IP proxy object and not the value directly,
        # so we query the value here
        if self._phase_object is not None:
            return [p.value for p in properties]
        else:
            return properties


class PlxStagedIPList:
    """
    Container-type object that maps a phase object to a PlxObjectPropertyList that
    iterates over the properties in the given phase.
    """

    def __init__(self, server, listable_parent, property_name):
        self._server = server
        self._listable_parent = listable_parent
        self._property_name = property_name

    def __getitem__(self, phase_object):
        # by defining __getitem__, the object can be iterated in a for loop with an integer key starting from 0.
        # if no exception is raised, the for loop is infinite. The following check solves this.
        if not isinstance(phase_object, plxproxy.PlxProxyObject):
            raise TypeError('Expected phase object key')

        return PlxObjectPropertyList(self._server, self._listable_parent, self._property_name, phase_object)

    def __repr__(self):
        return "<Staged Property list>"
