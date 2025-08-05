"""
Purpose: Create proxy objects from data supplied by the Plaxis HTTP API

Subversion data:
    $Id: plxproxyfactory.py 20186 2015-09-09 13:04:15Z tj $
    $URL: https://tools.plaxis.com/svn/sharelib/trunk/PlxObjectLayer/Server/plxscripting/plxproxyfactory.py $

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

from .plxproxy import (PlxProxyObject, PlxProxyObjectMethod,
                       PlxProxyGlobalObject, PlxProxyObjectProperty,
                       PlxProxyListable, PlxProxyValues, PlxProxyIPBoolean,
                       PlxProxyIPInteger, PlxProxyIPDouble, PlxProxyMaterial,
                       PlxProxyIPObject, PlxProxyIPText, 
                       PlxProxyIPEnumeration, PlxProxyIPStaged)

TYPE_BOOLEAN = "Boolean"
TYPE_TEXT = "Text"
TYPE_NUMBER = "Number"
TYPE_INTEGER = "Integer"
TYPE_OBJECT = "Object"
TYPE_ENUMERATION = "Enumeration"

ENUM = "enum"
STAGED = "staged"

from .const import (JSON_SUCCESS, JSON_ENUMVALUES, JSON_QUERIES, JSON_EXTRAINFO)


def is_primitive(plx_type):
    """Returns boolean which indicates whether the input plx_type is a primitive
    arguments:
        plx_type -- string indicating the type"""
    primitives = [TYPE_BOOLEAN, TYPE_NUMBER, TYPE_INTEGER, TYPE_TEXT]
    return plx_type in primitives or plx_type.startswith(ENUM)


class PlxProxyFactory():
    """
    Responsible for creation of proxy objects based on data supplied from HTTP
    API responses
    """

    def __init__(self, connection):
        """"
        Store the mixin class to avoid creating the same class for every
        listable.
        """
        self._connection = connection

        self.PlxProxyObjectListable = self.mix_in(
            PlxProxyObject, PlxProxyListable, 'PlxProxyObjectListable')

        self.PlxProxyObjectValues = self.mix_in(
            PlxProxyObject, PlxProxyValues, 'PlxProxyObjectValues')

        self.PlxProxyObjectMaterial = self.mix_in(
            PlxProxyObject, PlxProxyMaterial, 'PlxProxyObjectMaterial')

        self.PlxProxyObjectPropertyListable = self.mix_in(
            PlxProxyObjectProperty, PlxProxyListable, 'PlxProxyObjectPropertyListable')

        self.PlxProxyIPObjectListable = self.mix_in(
            PlxProxyIPObject, PlxProxyListable, 'PlxProxyIPObjectListable')

        self.proxy_object_cache = {} # Maps GUIDs to proxies
        self._proxy_enum_classes = {} # Maps enum names to enum classes

    def clear_proxy_object_cache(self):
        # Do NOT call this method if it's not truly needed (on account of
        # the cache having been invalidated by e.g. starting a new project).
        # There is functionality that relies on the cache being in a
        # correct state, in particular the way we deal with intrinsic
        # properties returned by a request other than 'members'.
        self.proxy_object_cache = {}

    def get_proxy_object_if_exists(self, guid):
        if guid in self.proxy_object_cache:
            return self.proxy_object_cache[guid]
        else:
            return None

    def create_plx_proxy_object(self, server, guid, plx_type, is_listable,
                                property_name='', owner=None):
        """
        Creates a new PlxProxyObject with the supplied guid and object type
        """
        if guid in self.proxy_object_cache:
            return self.proxy_object_cache[guid]

        if owner is not None:
            proxy_object = self._create_plx_proxy_property(
                server, guid, plx_type, is_listable, property_name, owner)
        elif plx_type == 'PlxValues':
            proxy_object = self.PlxProxyObjectValues(server, guid, plx_type)
        elif plx_type == 'SoilMat':
            proxy_object = self.PlxProxyObjectMaterial(server, guid, plx_type)
        elif is_listable:
            proxy_object = self.PlxProxyObjectListable(server, guid, plx_type)
        else:
            proxy_object = PlxProxyObject(server, guid, plx_type)

        self.proxy_object_cache[guid] = proxy_object

        return proxy_object

    def create_plx_proxy_object_method(self, server, proxy_object,
                                       method_name):
        """ Creates a new PlxProxyObjectMethod with the supplied name """
        proxy_method = PlxProxyObjectMethod(server, proxy_object, method_name)
        return proxy_method

    def create_plx_proxy_global(self, server):
        """ Creates a global proxy object """
        proxy_global = PlxProxyGlobalObject(server)
        return proxy_global

    def _create_plx_proxy_property(self, server, guid, plx_type, is_listable,
                                   property_name, owner):
        """ Creates a new PlxProxyObjectProperty """
        if plx_type == TYPE_BOOLEAN:
            proxy_property = PlxProxyIPBoolean(server, guid, plx_type,
                                               property_name, owner)
        elif plx_type.startswith(ENUM):
            proxy_enum_class = self._create_proxy_enumeration(guid, plx_type)
            proxy_property = proxy_enum_class(
                server, guid, plx_type, property_name, owner)
        elif plx_type == TYPE_NUMBER:
            proxy_property = PlxProxyIPDouble(server, guid, plx_type,
                                              property_name, owner)
        elif plx_type == TYPE_INTEGER:
            proxy_property = PlxProxyIPInteger(server, guid, plx_type,
                                               property_name, owner)
        elif plx_type == TYPE_OBJECT:
            if is_listable:
                proxy_property = self.PlxProxyIPObjectListable(
                    server, guid, plx_type, property_name, owner)
            else:
                proxy_property = PlxProxyIPObject(server, guid, plx_type,
                                                  property_name, owner)
        elif plx_type == TYPE_TEXT:
            proxy_property = PlxProxyIPText(server, guid, plx_type,
                                            property_name, owner)
        elif plx_type.startswith(STAGED):
            proxy_property = PlxProxyIPStaged(server, guid, plx_type,
                                              property_name, owner)
        else:
            if is_listable:
                proxy_property = self.PlxProxyObjectPropertyListable(
                    server, guid, plx_type, property_name, owner)
            else:
                proxy_property = PlxProxyObjectProperty(server, guid, plx_type,
                    property_name, owner)

        return proxy_property

    def mix_in(self, TargetClass, MixInClass, name=None):
        """
        Defines and returns a class object which inherits from both class
        objects that are supplied.
        """
        if name is None:
            name = TargetClass.__name__ + MixInClass.__name__

        # It appears to be important for the simpler mixin class to be the
        # first parameter here when mixing with a regular proxy object,
        # otherwise its constructor is not called. (And then, for instance, the
        # 'self.index' attribute of a listable mixin will not be set). This is
        # to do with the method resolution order in multiple inheritance.
        class CombinedClass(MixInClass, TargetClass):
            pass

        CombinedClass.__name__ = name
        return CombinedClass

    def _create_proxy_enumeration(self, proxy_enum_guid, proxy_enum_name):
        """
        Returns a class of the specified enumeration name.
        """
        if proxy_enum_name in self._proxy_enum_classes:
            return self._proxy_enum_classes[proxy_enum_name]

        response = self._connection.request_enumeration(
            proxy_enum_guid)[JSON_QUERIES][proxy_enum_guid]

        enum_dict = self._handle_enumeration_request(response)

        class PlxProxyIPEnumerationLocal(PlxProxyIPEnumeration):
            pass

        for key, val in enum_dict.items():
            # Python 2.7 doesn't allow non-ascii chars to be used in attribute names.
            sanitized_key = ''.join(c for c in key if ord(c) < 128)
            setattr(PlxProxyIPEnumerationLocal, sanitized_key, val)

        PlxProxyIPEnumerationLocal.__name__ = proxy_enum_name

        self._proxy_enum_classes[proxy_enum_name] = PlxProxyIPEnumerationLocal

        return PlxProxyIPEnumerationLocal

    def _handle_enumeration_request(self, enumeration_response):
        is_successful = enumeration_response[JSON_SUCCESS]
        if is_successful:
            return enumeration_response[JSON_ENUMVALUES]
        else:
            raise Exception(enumeration_response[JSON_EXTRAINFO])
