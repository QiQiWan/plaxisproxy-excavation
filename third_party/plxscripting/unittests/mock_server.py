"""
Purpose: Mock server.py module for the unittests

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

from plxscripting.plxproxy import PlxProxyObject, PlxProxyObjectProperty, PlxProxyListable
import plxscripting.const as const


class MockProxyObject(PlxProxyObject):

    def __init__(self, server, guid):
        super(MockProxyObject, self).__init__(server, guid, 'MockProxyObject')


class MockProxyObjectProperty(PlxProxyObjectProperty):

    def __init__(self, server, guid, owner, name):
        super(MockProxyObjectProperty, self).__init__(server, guid, 'MockProxyObjectProperty', name, owner)


class Server:

    def __init__(self):
        self._function_call_count = 0
        self.proxies_to_reset = []
        self.object_property_store = None
        self.object_property_phase_store = None
        self.object_property_to_send = 'uninitialized'
        self.listable_count = 5

    def log_function_call(self):
        self._function_call_count += 1

    def func_calls_since_last_check(self):
        old = self._function_call_count
        self._function_call_count = 0
        return old

    def get_name_by_guid(self, guid):
        return guid

    def get_named_object(self, object_name):
        return MockProxyObject(self, object_name)

    def get_object_attributes(self, obj):
        attributes = {
            'test_attribute': 12345,
            'UserFeatures': '',  # MockProxyObject(self, 'abcde'),
            'test_property_except_listable': MockProxyObjectProperty(self, 'mock_property_guid', obj, 'test_property_except_listable'),
            'delete': self.log_function_call
        }

        if isinstance(obj, PlxProxyListable):
            del attributes['test_property_except_listable']

        return attributes

    def get_object_property(self, proxy_object, prop_name, phase_object=None):
        return self.get_objects_property([proxy_object], prop_name, phase_object)[0]

    def get_objects_property(self, proxy_objects, prop_name, phase_object=None):
        self.object_property_phase_store = phase_object
        return [self.object_property_to_send] * len(proxy_objects)

    def set_object_property(self, proxy_property, prop_value):
        self.object_property_store = prop_value
        return True

    def call_listable_method(self, proxy_listable, method_name, startindex=None, stopindex=None, property_name=None):
        if method_name == const.COUNT:
            return self.listable_count
        elif method_name == const.SUBLIST:
            return ['mock_call_listable_method::command:{}::index:{}'.format(method_name, i) for i in
                    range(startindex, stopindex)]
        elif method_name == const.INDEX:
            return 'mock_call_listable_method::command:{}::index:{}'.format(method_name, startindex)
        elif method_name == const.MEMBERSUBLIST:
            return ['mock_call_listable_method::command:{}::index:{}::member_names:{}'.format(method_name, i,
                                                                                              [property_name]) for i in
                    range(startindex, stopindex)]
        elif method_name == const.MEMBERINDEX:
            return 'mock_call_listable_method::command:{}::index:{}::member_names:{}'.format(method_name, startindex,
                                                                                             [property_name])
        else:
            return 'mock_call_listable_method::command:{}'.format(method_name)

    def call_plx_object_method(self, proxy_obj, method_name, params):
        return '{}.{}({})'.format(proxy_obj, method_name, ', '.join([repr(p) for p in params]))

    def call_selection_command(self, command, *args):
        return [str(a) for a in args]

    def add_proxy_to_reset(self, proxy_obj):
        self.proxies_to_reset.append(proxy_obj)
