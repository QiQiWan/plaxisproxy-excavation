"""
Purpose: Unit tests for the selection.py module

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
from .. import plxproxyfactory, server
from ..selection import Selection


class fake_proxy_obj:
    def __init__(self, name, guid):
        self.__name__ = name
        self._guid = guid

    def get_cmd_line_repr(self):
        return self.__name__


@pytest.fixture
def get_prerequisites():
    connection = mock_connection.HTTPConnection('fake_host', 12345)
    _server = server.Server(connection, plxproxyfactory.PlxProxyFactory(connection), server.InputProcessor())
    selection = Selection(_server)
    return connection, _server, selection


def test_refresh(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    connection._selection = [None] * 5
    assert selection._objects == []
    selection.refresh()
    assert len(selection._objects) == 5


def test_set(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)
    assert connection._selection[0]['guid'] == 'guid1'
    assert connection._selection[1]['guid'] == 'guid2'


def test_append(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    selection.append(fake_proxy_obj('etc', 'guid3'), fake_proxy_obj('...', 'guid4'))
    assert connection._selection[0]['guid'] == 'guid3'
    assert connection._selection[1]['guid'] == 'guid4'

    selection.append(fake_proxy_obj('bla', 'guid5'), fake_proxy_obj('~~~', 'guid6'))
    assert connection._selection[0]['guid'] == 'guid3'
    assert connection._selection[1]['guid'] == 'guid4'
    assert connection._selection[2]['guid'] == 'guid5'
    assert connection._selection[3]['guid'] == 'guid6'


def test_extend(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)

    selection.extend([fake_proxy_obj('....', 'guid5'), fake_proxy_obj('.....', 'guid6')])
    assert connection._selection[0]['guid'] == 'guid1'
    assert connection._selection[1]['guid'] == 'guid2'
    assert connection._selection[2]['guid'] == 'guid5'
    assert connection._selection[3]['guid'] == 'guid6'


def test_remove(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)
    selection.extend([fake_proxy_obj('....', 'guid5'), fake_proxy_obj('.....', 'guid6')])

    selection.remove(*objlist)
    assert connection._selection[0]['guid'] == 'guid5'
    assert connection._selection[1]['guid'] == 'guid6'


def test_pop(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)
    selection.pop()
    assert len(connection._selection) == 1
    assert connection._selection[0]['guid'] == 'guid1'


def test_clear(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)
    selection.clear()
    assert connection._selection == []


def test_magic_methods(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    assert len(selection) == 0
    connection._selection = [None] * 5
    selection.refresh()
    assert len(selection) == 5


def test_magic_add_list_get(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    connection._selection = []
    selection.refresh()

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection += objlist
    assert len(selection) == 2
    assert selection[0]._guid == 'guid1'
    assert selection[1]._guid == 'guid2'

    with pytest.raises(IndexError):
        selection[1234]


def test_magic_add_obj(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)

    selection = selection + fake_proxy_obj('etc', 'guid3')
    assert len(selection) == 3
    assert selection[0]._guid == 'guid1'
    assert selection[1]._guid == 'guid2'
    assert selection[2]._guid == 'guid3'

    with pytest.raises(AttributeError) as exc:
        selection += 'not_a_proxy_object'
    assert "has no attribute '_guid'" in str(exc.value)


def test_magic_sub(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)

    selection -= selection._objects[0]
    assert len(selection) == 1
    assert selection[0]._guid == 'guid2'

    with pytest.raises(AttributeError):
        selection -= 'not_in_the_list'


def test_magic_setitem(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)

    selection[1] = fake_proxy_obj('...', 'guid4')
    assert len(selection) == 2
    assert selection[0]._guid == 'guid1'
    assert selection[1]._guid == 'guid4'


def test_magic_contains(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)

    assert selection[1] in selection
    assert 'cheese' not in selection


def test_magic_delitem(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)

    del selection[0]
    assert len(selection) == 1
    assert selection[0]._guid == 'guid2'


def test_magic_repr(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]

    selection += objlist
    assert repr(selection) == '[<MockItem_Selection guid1>, <MockItem_Selection guid2>]'


def test_magic_iter(get_prerequisites):
    connection, server_instance, selection = get_prerequisites

    objlist = [fake_proxy_obj('stuff', 'guid1'), fake_proxy_obj('things', 'guid2')]
    selection.set(objlist)

    for item in selection:
        assert 'guid' in item._guid
