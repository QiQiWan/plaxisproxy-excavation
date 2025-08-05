"""
Purpose: Unit tests for the server.py module

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
from . import mock_connection
from .. import server, plxproxyfactory, plxproxy, tokenizer
from ..selection import Selection
from ..plx_scripting_exceptions import PlxScriptingTokenizerError, PlxScriptingError


def newsrv():
    con = mock_connection.HTTPConnection('fake_host', 12345)
    return con, server.Server(con, plxproxyfactory.PlxProxyFactory(con), server.InputProcessor())


class fake_proxy_obj:
    def __init__(self, name, guid):
        self.__name__ = name
        self._guid = guid
        self._plx_type = 'MockItem_UnitTest'

    def get_cmd_line_repr(self):
        return self.__name__


def test_InputProcessor_param_to_string():
    obj = server.InputProcessor()

    # Simple tests
    assert obj.param_to_string('test') == '"test"'
    assert obj.param_to_string(54321) == '54321'
    assert obj.param_to_string([54321]) == '(54321)'
    assert obj.param_to_string((54321,)) == '(54321)'
    assert obj.param_to_string([54321, 'cheese']) == '(54321 "cheese")'
    assert obj.param_to_string((54321, 'cake')) == '(54321 "cake")'

    # Generator object
    def generator(n=5):
        for x in range(1, n + 1):
            yield x

    assert obj.param_to_string(generator()) == '(1 2 3 4 5)'

    # Selection object
    sel = Selection('i_am_a_fake_server_object')
    sel._objects = [1, 2, 3, 4, 5]
    assert obj.param_to_string(sel) == '(1 2 3 4 5)'

    class RandomObject:
        def __str__(self):
            return 'random_object'

    # fallback behaviour is to str() the input
    assert obj.param_to_string(RandomObject()) == 'random_object'


def test_InputProcessor_create_method_call_cmd():
    obj = server.InputProcessor()

    # Simple list
    resp2 = obj.create_method_call_cmd(fake_proxy_obj('fakeobj', 'fake_guid'), '_fakefunc', [12345])
    assert resp2 == '_fakefunc fakeobj 12345'

    # Mixed list
    resp = obj.create_method_call_cmd(fake_proxy_obj('Object_1_2', 'fake_guid'), 'function',
                                      ['param1', 'param2', 'param3', 4, 5])
    assert resp == 'function Object_1_2 "param1" "param2" "param3" 4 5'

    # No object
    resp2 = obj.create_method_call_cmd(None, '_fakefuncnoobj', [54321])
    assert resp2 == '_fakefuncnoobj 54321'


def test_ResultHandler_handle_namedobjects_response():
    con, srv = newsrv()
    pf = plxproxyfactory.PlxProxyFactory(con)
    obj = server.ResultHandler(srv, pf)

    input_ok = {'extrainfo': '',
                'success': True,
                'returnedobject': {
                    'islistable': True,
                    'type': 'ModelGroup',
                    'guid': '{5B0FCDB3-F02B-4356-AC82-683628DF3555}'
                }}

    response_ok = '<ModelGroup {5B0FCDB3-F02B-4356-AC82-683628DF3555}>'

    input_fail = {'extrainfo': 'Named object does not exist in current registry.',
                  'success': False,
                  'returnedobject': {}}

    response_fail = 'Named object does not exist in current registry.'

    result = obj.handle_namedobjects_response(input_ok)
    assert isinstance(result, plxproxy.PlxProxyObject)
    assert result._plx_type == 'ModelGroup'
    assert result._guid == '{5B0FCDB3-F02B-4356-AC82-683628DF3555}'

    with pytest.raises(PlxScriptingError) as exc:
        obj.handle_namedobjects_response(input_fail)
    assert response_fail in str(exc.value)


def test_ResultHandler_handle_commands_response():
    con, srv = newsrv()
    pf = plxproxyfactory.PlxProxyFactory(con)
    obj = server.ResultHandler(srv, pf)

    input_ok = {'extrainfo': 'Added Borehole_1',
                'returnedobjects': [{
                    'islistable': True,
                    'type': 'Borehole',
                    'guid': '{2A3F1B63-A6F2-48B0-9D3A-2A8F39E6D2D9}'}],
                'debuginfo': '',
                'success': True,
                'errorpos': -1,
                'returnedvalues': []}

    response_ok = '<Borehole {2A3F1B63-A6F2-48B0-9D3A-2A8F39E6D2D9}>'

    input_fail = {'extrainfo': 'Cannot intersect unless there is at least one volume or surface in the geometry',
                  'debuginfo': '',
                  'success': False,
                  'errorpos': -1}

    response_fail = 'Cannot intersect unless there is at least one volume or surface in the geometry'

    result = obj.handle_commands_response(input_ok)
    assert isinstance(result, plxproxy.PlxProxyObject)
    assert result._plx_type == 'Borehole'
    assert result._guid == '{2A3F1B63-A6F2-48B0-9D3A-2A8F39E6D2D9}'

    with pytest.raises(PlxScriptingError) as exc:
        obj.handle_commands_response(input_fail)
    assert response_fail in str(exc.value)


def test_ResultHandler_handle_members_response():
    con, srv = newsrv()
    pf = plxproxyfactory.PlxProxyFactory(con)
    obj = server.ResultHandler(srv, pf)

    input_ok = {'extrainfo': '',
                'success': True,
                'properties': {
                    'TypeName': {
                        'islistable': False,
                        'value': 'ModelGroup',
                        'type': 'Text',
                        'guid': '{52F2A950-7AAC-4413-8FCE-E9AEE3287CF1}',
                        'ispublished': False,
                        'ownerguid': '{AE5F2EF3-D5EA-4F46-84B2-909061DFD5A4}',
                        'caption': 'TypeName'},
                    'Name': {
                        'islistable': False,
                        'value': 'Boreholes',
                        'type': 'Text',
                        'guid': '{CF6CB8A7-1944-4967-A789-3C744BF339D3}',
                        'ispublished': False,
                        'ownerguid': '{AE5F2EF3-D5EA-4F46-84B2-909061DFD5A4}',
                        'caption': 'Name'},
                    'UserFeatures': {
                        'islistable': False,
                        'value': {
                            'islistable': True,
                            'type': 'PlxUserFeatureList',
                            'guid': '{C325B1E0-809F-4113-8F78-1037098857B6}'},
                        'type': 'Object',
                        'guid': '{E8F956DD-6F67-4DC7-96F5-AB31573D4336}',
                        'ispublished': False,
                        'ownerguid': '{AE5F2EF3-D5EA-4F46-84B2-909061DFD5A4}',
                        'caption': 'UserFeatures'},
                    'Comments': {
                        'islistable': False,
                        'value': '',
                        'type': 'Text',
                        'guid': '{9A5D6095-1D2A-49C6-BD94-6B32DC4465C1}',
                        'ispublished': False,
                        'ownerguid': '{AE5F2EF3-D5EA-4F46-84B2-909061DFD5A4}',
                        'caption': 'Comments'}
                },
                'commands': [
                    'echo',
                    '__dump',
                    'commands',
                    'multiply',
                    'info',
                    '__observers',
                    'setproperties',
                    'setmaterial'],
                'commandlinename': 'Boreholes'}
    result = obj.handle_members_response(input_ok, fake_proxy_obj('fake_name', 'fake_guid'))

    assert 'Name' in result
    assert result['Name']._guid == '{CF6CB8A7-1944-4967-A789-3C744BF339D3}'
    assert 'UserFeatures' in result
    assert hasattr(result['UserFeatures'], '__iter__')

    # This function has no error handler, so there is no failure to test.


def test_ResultHandler_handle_list_response():
    con, srv = newsrv()
    pf = plxproxyfactory.PlxProxyFactory(con)
    obj = server.ResultHandler(srv, pf)

    # count
    input_ok = {'extrainfo': '',
                'success': True,
                'methodname': 'count',
                'guid': '{8D251E47-C14C-4A08-8506-4D5CE6C316CF}',
                'outputdata': 3}
    response = 3

    result = obj.handle_list_response(input_ok)
    assert result == 3

    # index
    input_ok = {'extrainfo': '',
                'success': True,
                'startindex': 0,
                'methodname': 'index',
                'guid': '{F9A93D18-4FB2-4A74-A564-C2F25D9814FB}',
                'outputdata': {
                    'islistable': True,
                    'type': 'Line',
                    'guid': '{6D4BAEF4-E8EF-45B0-B796-E86F51D02DFB}'}}

    result = obj.handle_list_response(input_ok)
    assert isinstance(result, plxproxy.PlxProxyObject)
    assert result._plx_type == 'Line'

    # sublist
    input_ok = {'extrainfo': '',
                'stopindex': 3,
                'success': True,
                'startindex': 0,
                'methodname': 'sublist',
                'guid': '{28444FD3-7BC6-4039-9264-D605EC85C2CA}',
                'outputdata': [
                    {'islistable': False,
                     'type': 'Soil',
                     'guid': '{D535EAB7-8498-491E-B655-F954B34B2C33}'},
                    {'islistable': True,
                     'type': 'PorePressure',
                     'guid': '{CBCB4813-098F-4278-A48A-C4CCA78EECDA}'},
                    {'islistable': True,
                     'type': 'LayerZone',
                     'guid': '{F2C65547-8B8F-41C1-BEA7-0C4A1D105B50}'}]
                }

    result = obj.handle_list_response(input_ok)
    assert len(result) == 3
    assert isinstance(result[0], plxproxy.PlxProxyObject)
    assert result[0]._plx_type == 'Soil'
    assert isinstance(result[1], plxproxy.PlxProxyObject)
    assert result[1]._plx_type == 'PorePressure'
    assert isinstance(result[2], plxproxy.PlxProxyObject)
    assert result[2]._plx_type == 'LayerZone'

    # This function has no error handler, so there is no failure to test.


def test_ResultHandler_handle_propertyvalues_response():
    con, srv = newsrv()
    pf = plxproxyfactory.PlxProxyFactory(con)
    obj = server.ResultHandler(srv, pf)

    input1_ok = {'extrainfo': '',
                'success': True,
                'properties': {
                    'UserFeatures': {
                        'islistable': True,
                        'type': 'PlxUserFeatureList',
                        'guid': '{C325B1E0-809F-4113-8F78-1037098857B6}'}
                }}

    input2_ok = {'extrainfo': '',
                'success': True,
                'properties': {
                    'UserFeatures': {
                        'islistable': True,
                        'type': 'PlxUserFeatureList',
                        'guid': '{C0CA551D-DFD7-4FD0-AB3B-BAAD739BC820}'}
                }}

    input_fail = {'extrainfo': 'Fake error :)',
                  'success': False,
                  'properties': {}}

    [result1, result2] = obj.handle_propertyvalues_response([input1_ok, input2_ok], 'UserFeatures', 'ModelGroup')
    assert isinstance(result1, plxproxy.PlxProxyObject)
    assert isinstance(result2, plxproxy.PlxProxyObject)
    assert result1._plx_type == 'PlxUserFeatureList'
    assert result2._plx_type == 'PlxUserFeatureList'
    assert result1._guid == '{C325B1E0-809F-4113-8F78-1037098857B6}'
    assert result2._guid == '{C0CA551D-DFD7-4FD0-AB3B-BAAD739BC820}'

    result = obj.handle_propertyvalues_response([input_fail], 'FakeAttribute', 'FakeType')[0]
    assert result is None

    # Should also fail if one of the responses is failure
    result = obj.handle_propertyvalues_response([input1_ok, input_fail], 'FakeAttribute', 'FakeType')
    for r in result:
        assert r is None

def test_ResultHandler_handle_selection_response():
    con, srv = newsrv()
    pf = plxproxyfactory.PlxProxyFactory(con)
    obj = server.ResultHandler(srv, pf)

    input_ok = {'ReplyCode': '00000000000000000000000000000000',
                'selection': [
                    {'islistable': True,
                     'type': 'Line',
                     'guid': '{27498209-C64B-476B-8583-97233801C0DB}'},
                    {'islistable': True,
                     'type': 'Line',
                     'guid': '{5E480CA7-001C-4435-9888-1CFC029CD48E}'},
                    {'islistable': True,
                     'type': 'Line',
                     'guid': '{9430A8A8-DC90-4006-8EBE-0835799BB1F1}'}]
                }

    input_empty = {'ReplyCode': '00000000000000000000000000000000', 'selection': []}

    result = obj.handle_selection_response(input_ok)
    assert isinstance(result, list)
    assert len(result) == 3
    assert isinstance(result[0], plxproxy.PlxProxyObject)
    assert result[0]._guid == '{27498209-C64B-476B-8583-97233801C0DB}'
    assert result[0]._plx_type == 'Line'
    assert isinstance(result[1], plxproxy.PlxProxyObject)
    assert result[1]._guid == '{5E480CA7-001C-4435-9888-1CFC029CD48E}'
    assert result[1]._plx_type == 'Line'
    assert isinstance(result[2], plxproxy.PlxProxyObject)
    assert result[2]._guid == '{9430A8A8-DC90-4006-8EBE-0835799BB1F1}'
    assert result[2]._plx_type == 'Line'

    result = obj.handle_selection_response(input_empty)
    assert result == []


def test_Server_new():
    con, srv = newsrv()

    # Artificially put something in the caches
    srv._Server__globals_cache = {'a': 'b'}
    srv._Server__values_cache = {'a': 'b'}
    srv._Server__listables_cache = {'a': 'b'}
    srv._Server__proxy_factory.proxy_object_cache = {'a': 'b'}

    assert srv.new() == 'OK'

    # Make sure the caches are cleared
    assert srv._Server__globals_cache == {}
    assert srv._Server__values_cache == {}
    assert srv._Server__listables_cache == {}
    assert srv._Server__proxy_factory.proxy_object_cache == {}


def test_Server_recover():
    con, srv = newsrv()

    # Artificially put something in the caches
    srv._Server__globals_cache = {'a': 'b'}
    srv._Server__values_cache = {'a': 'b'}
    srv._Server__listables_cache = {'a': 'b'}
    srv._Server__proxy_factory.proxy_object_cache = {'a': 'b'}

    assert srv.recover() == 'OK'

    # Make sure the caches are cleared
    assert srv._Server__globals_cache == {}
    assert srv._Server__values_cache == {}
    assert srv._Server__listables_cache == {}
    assert srv._Server__proxy_factory.proxy_object_cache == {}


def test_Server_open():
    con, srv = newsrv()

    # Artificially put something in the caches
    srv._Server__globals_cache = {'a': 'b'}
    srv._Server__values_cache = {'a': 'b'}
    srv._Server__listables_cache = {'a': 'b'}
    srv._Server__proxy_factory.proxy_object_cache = {'a': 'b'}

    assert srv.open('filepath') == 'OK'

    # Make sure the caches are cleared
    assert srv._Server__globals_cache == {}
    assert srv._Server__values_cache == {}
    assert srv._Server__listables_cache == {}
    assert srv._Server__proxy_factory.proxy_object_cache == {}


def test_Server_close():
    con, srv = newsrv()

    # Artificially put something in the caches
    srv._Server__globals_cache = {'a': 'b'}
    srv._Server__values_cache = {'a': 'b'}
    srv._Server__listables_cache = {'a': 'b'}
    srv._Server__proxy_factory.proxy_object_cache = {'a': 'b'}

    assert srv.close() == 'OK'

    # Make sure the caches are cleared
    assert srv._Server__globals_cache == {}
    assert srv._Server__values_cache == {}
    assert srv._Server__listables_cache == {}
    assert srv._Server__proxy_factory.proxy_object_cache == {}


def test_Server_call_listable_method():
    con, srv = newsrv()

    guid = 'mock_guid'
    obj = fake_proxy_obj('name', guid)

    resp = srv.call_listable_method(obj, 'count')
    assert resp == 3

    resp = srv.call_listable_method(obj, 'index', startindex=2)
    assert mock_connection.MOCK_GUID_PREFIX in resp._guid
    assert 'MockItem_' in resp._plx_type

    resp = srv.call_listable_method(obj, 'sublist', startindex=3, stopindex=6)
    assert len(resp) == 3
    for r in resp:
        assert mock_connection.MOCK_GUID_PREFIX in r._guid
        assert 'MockItem_' in r._plx_type


def test_Server_get_named_object():
    con, srv = newsrv()

    resp = srv.get_named_object('fake_name')
    assert mock_connection.MOCK_GUID_PREFIX in resp._guid
    assert 'MockItem_' in resp._plx_type


def test_Server_get_object_property():
    con, srv = newsrv()
    obj = fake_proxy_obj('fake_name', 'fake_guid')

    resp = srv.get_object_property(obj, 'Name')
    assert resp == 'MockItem_Name_pval'

    resp = srv.get_object_property(obj, 'UserFeatures')
    assert mock_connection.MOCK_GUID_PREFIX in resp._guid
    assert 'MockItem_' in resp._plx_type

def test_Server_get_objects_property():
    con, srv = newsrv()
    obj1 = fake_proxy_obj('fake_name_1', 'fake_guid_1')
    obj2 = fake_proxy_obj('fake_name_2', 'fake_guid_2')

    [resp1, resp2] = srv.get_objects_property([obj1, obj2], 'Name')
    assert resp1 == 'MockItem_Name_pval'
    assert resp2 == 'MockItem_Name_pval'

    [resp1, resp2] = srv.get_objects_property([obj1, obj2], 'UserFeatures')
    assert mock_connection.MOCK_GUID_PREFIX in resp1._guid
    assert mock_connection.MOCK_GUID_PREFIX in resp2._guid
    assert 'MockItem_' in resp1._plx_type
    assert 'MockItem_' in resp2._plx_type

def test_Server_get_name_by_guid():
    con, srv = newsrv()
    assert srv.get_name_by_guid('fake_guid') == 'MockItem_Name_pval'


def test_Server_get_object_attributes():
    con, srv = newsrv()

    obj = fake_proxy_obj('fake_name', 'fake_guid')

    resp = srv.get_object_attributes(obj)
    assert isinstance(resp['Mock_number'], plxproxy.PlxProxyIPNumber)
    assert isinstance(resp['Name'], plxproxy.PlxProxyIPText)
    assert isinstance(resp['IsDynamicComponent'], plxproxy.PlxProxyIPBoolean)


def test_Server_call_commands():
    con, srv = newsrv()

    resp = srv.call_commands('Command 1', 'Command 2')
    assert resp[0]['feedback']['success'] == True
    assert resp[0]['feedback']['extrainfo'] == 'Reply_to_command: Command 1'
    assert resp[1]['feedback']['extrainfo'] == 'Reply_to_command: Command 2'

    con.test_success = False
    resp = srv.call_commands('gotomesh')
    assert resp[0]['feedback']['success'] == False
    assert resp[0]['feedback']['extrainfo'] == 'Cannot intersect unless there is at least one volume or surface in the geometry'


def test_Server_call_and_handle_commands():
    con, srv = newsrv()

    resp = srv.call_and_handle_commands('Command 1', 'Command 2', con.MAGIC_REQUEST_OBJECT)
    assert resp[0] == 'Reply_to_command: Command 1'
    assert resp[1] == 'Reply_to_command: Command 2'
    assert isinstance(resp[2], plxproxy.PlxProxyObject)


def test_Server_call_plx_object_method():
    con, srv = newsrv()

    obj = fake_proxy_obj('fake_object', 'fake_guid')

    resp = srv.call_plx_object_method(obj, 'method', ['param_1', 2, 'param_3'])
    assert resp == 'Reply_to_command: method fake_object "param_1" 2 "param_3"'

    resp = srv.call_plx_object_method(obj, con.MAGIC_REQUEST_OBJECT, ['param_1', 2, 'param_3'])
    assert isinstance(resp, plxproxy.PlxProxyObject)


def test_Server_set_object_property():
    con, srv = newsrv()

    obj = fake_proxy_obj('fake_name', 'fake_guid')
    assert srv.set_object_property(obj, ['fake_value_1', 2, 3]) == 'Reply_to_command: set fake_name "fake_value_1" 2 3'


def test_Server_call_selection_command():
    con, srv = newsrv()

    obj1 = fake_proxy_obj('obj_boter', '<boter>')
    obj2 = fake_proxy_obj('obj_kaas', '<kaas>')
    obj3 = fake_proxy_obj('obj_eieren', '<eieren>')

    # set + response data + get
    resp = srv.call_selection_command('set', obj1, obj2, obj3)
    assert len(resp) == 3
    assert resp[0]._guid == '<boter>'
    assert resp[1]._guid == '<kaas>'
    assert resp[2]._guid == '<eieren>'
    assert len(srv.call_selection_command('get')) == 3

    # remove specific items + get
    resp = srv.call_selection_command('remove', obj2)
    assert len(resp) == 2
    resp = srv.call_selection_command('get')
    assert resp[0]._guid == '<boter>'
    assert resp[1]._guid == '<eieren>'

    # append + get
    resp = srv.call_selection_command('append', obj2)
    assert len(resp) == 3
    resp = srv.call_selection_command('get')
    assert resp[0]._guid == '<boter>'
    assert resp[1]._guid == '<eieren>'
    assert resp[2]._guid == '<kaas>'

    # clear
    assert len(srv.call_selection_command('set')) == 0


def test_Server_get_error():
    con, srv = newsrv()
    assert srv.get_error() == mock_connection.MOCK_EXCEPTION
    assert srv.get_error() == ''
    con.test_exception_cleared = False
    assert srv.get_error(False) == mock_connection.MOCK_EXCEPTION
    assert srv.get_error() == mock_connection.MOCK_EXCEPTION


def test_Server_tokenize():
    con, srv = newsrv()

    resp = srv.tokenize('plaxis_command object_name "string_arg" 1 2 "string_arg2"')
    assert isinstance(resp, tokenizer.TokenizerResultHandler)
    assert resp.extrainfo == ''
    assert resp.success == True
    assert isinstance(resp.tokens[0], tokenizer.TokenIdentifier)
    assert resp.tokens[0].value == 'command'
    assert resp.tokens[0].position == 0
    assert resp.tokens[0].length == 7
    assert isinstance(resp.tokens[1], tokenizer.TokenOperand)
    assert resp.tokens[1].value == 'object'
    assert resp.tokens[1].position == 8
    assert resp.tokens[1].length == 6
    assert isinstance(resp.tokens[2], tokenizer.TokenText)
    assert resp.tokens[2].value == '"param"'
    assert resp.tokens[2].position == 15
    assert resp.tokens[2].length == 7
    assert isinstance(resp.tokens[4], tokenizer.TokenInteger)
    assert resp.tokens[4].value == 2
    assert resp.tokens[4].position == 32
    assert resp.tokens[4].length == 1

    con.test_success = False
    resp = srv.tokenize('1234 "bla bla 542 2')
    assert isinstance(resp, tokenizer.TokenizerResultHandler)
    assert resp.extrainfo == 'Unbalanced quotes'
    assert resp.success == False
    with pytest.raises(PlxScriptingTokenizerError) as exc:
        resp.tokens[0]
    assert str(exc.value) == 'Unrecognized token at position 5'
