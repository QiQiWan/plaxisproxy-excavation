"""
Purpose: Unit tests for the tokenizer.py module

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
from .. import server, plxproxyfactory
from ..plx_scripting_exceptions import PlxScriptingTokenizerError


def test_successful_tokenization():
    con = mock_connection.HTTPConnection('fake_host', 12345)
    s = server.Server(con, plxproxyfactory.PlxProxyFactory(con), server.InputProcessor())

    tokenizer = s.tokenize('addpoint Polygon_2 0 4.2 0 # comment')  # 'command object "param" "param2" 2 3 4'
    assert tokenizer.success
    assert tokenizer.error_position == -1
    assert tokenizer.tokens == tokenizer.partial_tokens

    tokens = tokenizer.tokens
    assert len(tokens) == 7

    token = tokens[0]
    assert token.position == 0
    assert token.length == 7
    assert token.end_position == 6
    assert token.value == 'command'

    assert tokens[2].value == '"param"'

    assert tokens[-1].value == 4


def test_unsuccessful_tokenization():
    con = mock_connection.HTTPConnection('fake_host', 12345)
    s = server.Server(con, plxproxyfactory.PlxProxyFactory(con), server.InputProcessor())

    con.test_success = False
    tokenizer = s.tokenize('abc ?')
    with pytest.raises(PlxScriptingTokenizerError) as exception_info:
        tokenizer.tokens
    assert str(exception_info.value) == 'Unrecognized token at position 5'
    assert not tokenizer.success
    assert tokenizer.error == 'Unrecognized token at position 5'
    assert tokenizer.error_position == 5
    assert len(tokenizer.partial_tokens) == 1  # Since the first token is OK


def test_external_interpreter():
    con = mock_connection.HTTPConnection('fake_host', 12345)
    s = server.Server(con, plxproxyfactory.PlxProxyFactory(con), server.InputProcessor())

    con.test_tokenize_external = True
    tokenizer = s.tokenize('/command object "param" "param2" 2 3 4')

    token = tokenizer.tokens[0]
    assert token.interpretername == 'command'
    assert token.externalcommand == 'object "param" "param2" 2 3 4'
    assert token.content == '/command object "param" "param2" 2 3 4'
