"""
Purpose: Unit tests for the plxproxy.py and plxproxyfactory.py modules

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

import pytest
from . import mock_server
from .. import plxobjects, plxproxy, plxproxyfactory, const
from ..plx_scripting_exceptions import PlxScriptingError


@pytest.fixture
def environment():
    GUID = '<fake_guid>'
    GUID_OWNER = '<fake_guid_owner>'
    PLX_TYPE = 'fake_type'
    server = mock_server.Server()
    owner = plxproxy.PlxProxyObject_Abstract(server, GUID_OWNER)
    return server, owner, GUID, GUID_OWNER, PLX_TYPE


def test_PlxProxyObject_Abstract(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyObject_Abstract(server, GUID)
    assert obj.get_cmd_line_repr() == GUID
    assert obj.get_equivalent()._guid == GUID
    assert obj.get_equivalent(owner)._guid == GUID_OWNER


def test_PlxProxyGlobalObject(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyGlobalObject(server)

    with pytest.raises(PlxScriptingError) as exc:
        obj.selection = 12345
    assert '12345' in str(exc.value)

    selection = ['fake_object_selection_1', 'fake_object_selection_2']
    obj.selection = selection
    assert list(obj._selection) == selection

    assert str(obj.selection) == '<MockProxyObject selection>'  # should try to get selection from server


def test_PlxProxyObject(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyObject(server, GUID, PLX_TYPE)
    assert PLX_TYPE in str(obj)
    assert GUID in str(obj)

    obj.custom_attribute = 54321
    assert hasattr(obj, 'custom_attribute')
    assert obj.custom_attribute == 54321

    assert hasattr(obj, 'test_attribute')
    assert obj.test_attribute == 12345

    # todo test userfeatures

    with pytest.raises(AttributeError):
        __ = obj.non_existent_attribute


def test_PlxProxyListable(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment

    class Mixer(plxproxy.PlxProxyListable, plxproxy.PlxProxyObject):
        pass

    obj = Mixer(server, GUID, PLX_TYPE)

    # set within range
    obj[0] = 1
    obj[4] = 5

    # set out of range
    with pytest.raises(IndexError):
        obj[server.listable_count]

    # get within range
    assert 'index::index:1' in obj[1]

    # get out of range
    with pytest.raises(IndexError):
        a = obj[5]

    # iterate with cache
    for indx, val in enumerate(obj):
        assert 'sublist::index:{}'.format(indx) in val
    assert indx == 4


def test_PlxObjectPropertyList(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment

    class Mixer(plxproxy.PlxProxyListable, plxproxy.PlxProxyObject):
        pass

    listable = Mixer(server, GUID, PLX_TYPE)
    listable[0] = owner

    # TODO: Write the test when the Mock objects use PlxProxyObjects
    #       instead of pre-defined primitive values


def test_PlxProxyValues(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment

    class Mixer(plxproxy.PlxProxyObject, plxproxy.PlxProxyValues):
        pass

    obj = Mixer(server, GUID, PLX_TYPE)

    with obj:
        pass

    # make sure delete was called on obj (once!). This function is injected in mock server's get_object_attributes.
    assert server.func_calls_since_last_check() == 1


def test_PlxProxyObjectMethod(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    parent = plxproxy.PlxProxyObject(server, GUID, PLX_TYPE)
    obj = plxproxy.PlxProxyObjectMethod(server, parent, 'sandwich')

    blt = f"<{PLX_TYPE} {GUID}>.sandwich('tomato', 'lettuce', 'bacon')"
    assert obj('tomato', 'lettuce', 'bacon') == blt


def test_PlxProxyMaterial(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyMaterial(server, GUID, PLX_TYPE)
    assert obj in server.proxies_to_reset


def test_PlxProxyObjectProperty(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyObjectProperty(server, GUID, PLX_TYPE, 'property', owner)

    class Temp:
        property = obj

    tmp = Temp()

    # test __set__
    tmp.property = 2222
    assert server.object_property_store[0] == 2222  # make sure it was sent to server
    server.object_property_to_send = [42]
    assert tmp.property.value[0] == 42  # make sure it was retrieved from server intact

    # test stagedIP value
    obj.set_stagedIP_value([67890])
    assert tmp.property.value[0] == 67890


def test_PlxProxyIPBoolean(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyIPBoolean(server, GUID, PLX_TYPE, 'boolean', owner)

    obj.set_stagedIP_value(True)
    assert obj.value is True

    assert obj is not True
    assert obj == True
    assert obj != False
    assert not (obj == False)
    assert not (obj != True)
    assert obj != None
    assert not (obj == None)


def test_PlxProxyIPNumber(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyIPNumber(server, GUID, PLX_TYPE, 'number', owner)
    server.object_property_to_send = 20
    assert obj + obj + 20 + 0.1 == 60.1
    # Should be mostly tested by test_PlxProxyIPInteger and test_PlxProxyIPDouble, this is just a unit-smoketest


def test_PlxProxyIPInteger(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyIPInteger(server, GUID, PLX_TYPE, 'integer', owner)
    server.object_property_to_send = 10

    assert obj.value == 10
    assert 11 + obj == 21
    assert obj + 11 == 21
    assert 200 / obj == 20
    assert obj / 2 == 5
    assert 5 * obj == 50
    assert obj * 5 == 50
    assert obj - 1 == 9
    assert 1 - obj == -9
    assert obj ** 2 == 100
    assert 2 ** obj == 1024
    assert obj % 3 == 1
    assert 19 % obj == 9
    assert 1 < obj
    assert obj > 1
    assert obj == 10
    assert obj >= 5
    assert obj >= 10
    assert obj <= 10
    assert obj <= 15
    assert obj != 11


def test_PlxProxyIPDouble(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyIPDouble(server, GUID, PLX_TYPE, 'double', owner)
    server.object_property_to_send = 10.1

    assert obj.value == 10.1
    assert 11 + obj == 21.1
    assert obj + 11 == 21.1
    assert 202 / obj == 20
    assert obj / 2 == 5.05
    assert 5 * obj == 50.5
    assert obj * 5 == 50.5
    assert obj - 1 == 9.1
    assert 1 - obj == -9.1
    assert round(obj ** 2, 2) == 102.01
    assert 1 ** obj == 1
    assert round(obj % 3, 2) == 1.1
    assert 19 % obj == 8.9
    assert 1 < obj
    assert obj > 1
    assert obj == 10.1
    assert obj >= 5
    assert obj >= 10.1
    assert obj <= 10.1
    assert obj <= 15
    assert obj != 11


def test_PlxProxyIPObject(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj_as_value = plxproxy.PlxProxyObject(server, GUID, PLX_TYPE)
    obj_as_value.attribute_1 = '1'
    obj_as_value.attribute_2 = '2'

    obj = plxproxy.PlxProxyIPObject(server, GUID, PLX_TYPE, 'ipobject', owner)
    obj.set_stagedIP_value(obj_as_value)
    assert obj.value == obj_as_value

    # Todo find out why this infinitely recurses
    # assert hasattr(obj, 'attribute_1')
    # assert hasattr(obj, 'attribute_2')


def test_PlxProxyIPEnumeration(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment

    class TstEnum(plxproxy.PlxProxyIPEnumeration):
        enum_a = 'a'
        enum_b = 'b'

    obj = TstEnum(server, GUID, PLX_TYPE, 'ipenum', owner)

    # attribute exists, value matches server
    server.object_property_to_send = 'a'
    obj.strvalue = 'enum_a'
    assert obj.strvalue == 'enum_a'

    # attribute does not exist in class
    server.object_property_to_send = 'd'
    with pytest.raises(ValueError):
        obj.strvalue = 'enum_d'

    # attribute value does not match server response
    server.object_property_to_send = 'z'
    with pytest.raises(ValueError):
        obj.strvalue = 'enum_b'


def test_PlxProxyIPText(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyIPText(server, GUID, PLX_TYPE, 'iptext', owner)
    server.object_property_to_send = 'abcd'
    assert obj == 'abcd'
    assert obj + 'efgh' == 'abcdefgh'
    assert obj[2] == 'c'
    assert obj != 'dcba'


def test_PlxProxyIPStaged(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = plxproxy.PlxProxyIPStaged(server, GUID, PLX_TYPE, 'ipstaged', owner)

    some_phase = plxproxy.PlxProxyObject(server, '<some_phase_guid>', 'Phase')
    another_phase = plxproxy.PlxProxyObject(server, '<another_phase_guid>', 'Phase')
    server.object_property_to_send = 'not_relevant'

    server.object_property_phase_store = None
    __ = obj[some_phase]
    assert server.object_property_phase_store == some_phase

    server.object_property_store = None
    obj[another_phase] = 'some_value'
    assert server.object_property_store == [another_phase, 'some_value']


def test_PlxProxyFactory_mix_in():
    connection = 'fake'
    obj = plxproxyfactory.PlxProxyFactory(connection)

    class a: pass

    class b: pass

    c = obj.mix_in(a, b, name='test_name')()
    assert isinstance(c, b)
    assert isinstance(c, a) and isinstance(c, b)
    assert type(c).__name__ == 'test_name'
    c = obj.mix_in(b, a)()
    assert isinstance(c, a) and isinstance(c, b)


def test_PlxProxyFactory_create_plx_proxy_global(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    connection = 'fake'
    obj = plxproxyfactory.PlxProxyFactory(connection)
    ret_obj = obj.create_plx_proxy_global(server)
    assert isinstance(ret_obj, plxproxy.PlxProxyGlobalObject)


def test_PlxProxyFactory_create_plx_proxy_object_method(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    connection = 'fake'
    obj = plxproxyfactory.PlxProxyFactory(connection)
    ret_obj = obj.create_plx_proxy_object_method(server, owner, 'cheese')
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectMethod)


def proxyfactory():
    class mock_connection:
        def request_enumeration(self, *args, **kwargs):
            return {const.JSON_QUERIES: {
                'test_guid_1_2': {
                    const.JSON_SUCCESS: True,
                    const.JSON_ENUMVALUES: {}
                }}}

    connection = mock_connection()
    return plxproxyfactory.PlxProxyFactory(connection)


def test_PlxProxyFactory_create_plx_proxy_object(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_1', 'test_plxtype', False, property_name='test_property',
                                          owner=owner)
    assert isinstance(ret_obj, plxproxy.PlxProxyObject)
    assert not isinstance(ret_obj, plxproxy.PlxProxyValues)
    assert not isinstance(ret_obj, plxproxy.PlxProxyMaterial)
    assert not isinstance(ret_obj, plxproxy.PlxProxyIPObject)
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert not isinstance(ret_obj, plxproxy.PlxProxyListable)


def test_PlxProxyFactory_create_PlxProxyIPBoolean(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_1_1', plxproxyfactory.TYPE_BOOLEAN, False,
                                          property_name='test_property', owner=owner)
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert isinstance(ret_obj, plxproxy.PlxProxyIPBoolean)


def test_PlxProxyFactory_create_PlxProxyIPEnumeration(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_1_2', plxproxyfactory.ENUM, False,
                                          property_name='test_property',
                                          owner=owner)
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert isinstance(ret_obj, plxproxy.PlxProxyIPEnumeration)


def test_PlxProxyFactory_create_PlxProxyIPDouble(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_1_3', plxproxyfactory.TYPE_NUMBER, False,
                                          property_name='test_property', owner=owner)
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert isinstance(ret_obj, plxproxy.PlxProxyIPDouble)


def test_PlxProxyFactory_create_PlxProxyIPInteger(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_1_4', plxproxyfactory.TYPE_INTEGER, False,
                                          property_name='test_property', owner=owner)
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert isinstance(ret_obj, plxproxy.PlxProxyIPInteger)


def test_PlxProxyFactory_create_PlxProxyObjectProperty_non_listable(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_1_5', plxproxyfactory.TYPE_OBJECT, False,
                                          property_name='test_property', owner=owner)
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert not isinstance(ret_obj, plxproxy.PlxProxyListable)


def test_PlxProxyFactory_create_PlxProxyListable(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_1_6', plxproxyfactory.TYPE_OBJECT, True,
                                          property_name='test_property', owner=owner)
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert isinstance(ret_obj, plxproxy.PlxProxyListable)


def test_PlxProxyFactory_create_PlxProxyIPStaged(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_1_7', plxproxyfactory.STAGED, False,
                                          property_name='test_property',
                                          owner=owner)
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert isinstance(ret_obj, plxproxy.PlxProxyIPStaged)


def test_PlxProxyFactory_create_PlxProxyListable_unknown_type(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_1_8', 'test_plxtype', True, property_name='test_property',
                                          owner=owner)
    assert isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert isinstance(ret_obj, plxproxy.PlxProxyListable)


def test_PlxProxyFactory_create_PlxProxyValues(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_2', 'PlxValues', False, property_name='test_property',
                                          owner=None)
    assert isinstance(ret_obj, plxproxy.PlxProxyObject)
    assert isinstance(ret_obj, plxproxy.PlxProxyValues)
    assert not isinstance(ret_obj, plxproxy.PlxProxyMaterial)
    assert not isinstance(ret_obj, plxproxy.PlxProxyIPObject)
    assert not isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert isinstance(ret_obj, plxproxy.PlxProxyListable)


def test_PlxProxyFactory_create_PlxProxyMaterial(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_3', 'SoilMat', False, property_name='test_property',
                                          owner=None)
    assert isinstance(ret_obj, plxproxy.PlxProxyObject)
    assert not isinstance(ret_obj, plxproxy.PlxProxyValues)
    assert isinstance(ret_obj, plxproxy.PlxProxyMaterial)
    assert not isinstance(ret_obj, plxproxy.PlxProxyIPObject)
    assert not isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert not isinstance(ret_obj, plxproxy.PlxProxyListable)


def test_PlxProxyFactory_create_PlxProxyListable_unknown_type_no_owner(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_uid_4', 'test_plxtype', True, property_name='test_property',
                                          owner=None)
    assert isinstance(ret_obj, plxproxy.PlxProxyObject)
    assert not isinstance(ret_obj, plxproxy.PlxProxyValues)
    assert not isinstance(ret_obj, plxproxy.PlxProxyMaterial)
    assert not isinstance(ret_obj, plxproxy.PlxProxyIPObject)
    assert not isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert isinstance(ret_obj, plxproxy.PlxProxyListable)


def test_PlxProxyFactory_create_PlxProxyObjectProperty_no_list_no_own(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    obj = proxyfactory()

    ret_obj = obj.create_plx_proxy_object(server, 'test_guid_5', 'test_plxtype', False, property_name='test_property',
                                          owner=None)
    assert isinstance(ret_obj, plxproxy.PlxProxyObject)
    assert not isinstance(ret_obj, plxproxy.PlxProxyValues)
    assert not isinstance(ret_obj, plxproxy.PlxProxyMaterial)
    assert not isinstance(ret_obj, plxproxy.PlxProxyIPObject)
    assert not isinstance(ret_obj, plxproxy.PlxProxyObjectProperty)
    assert not isinstance(ret_obj, plxproxy.PlxProxyListable)


def test_PlxProxyFactory_get_proxy_object_if_exists(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    connection = 'fake'
    obj = plxproxyfactory.PlxProxyFactory(connection)

    assert obj.get_proxy_object_if_exists('test_guid') is None
    obj.create_plx_proxy_object(server, 'test_guid', 'test_plxtype', False, property_name='test_property',
                                owner=None)
    assert obj.get_proxy_object_if_exists('test_guid') is not None


def test_PlxProxyFactory_clear_proxy_object_cache(environment):
    server, owner, GUID, GUID_OWNER, PLX_TYPE = environment
    connection = 'fake'
    obj = plxproxyfactory.PlxProxyFactory(connection)

    obj.create_plx_proxy_object(server, 'testymctestface', 'test_plxtype', False, property_name='test_property',
                                owner=None)
    assert obj.get_proxy_object_if_exists('testymctestface') is not None

    obj.clear_proxy_object_cache()
    assert obj.get_proxy_object_if_exists('testymctestface') is None
