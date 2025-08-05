"""
Purpose: Tokenizing is the process of splitting a string into substrings of specific types (tokens) understood by the
command line in PLAXIS applications. E.g. a token may end up being interpreted as an object name, a numerical value,
etc. Tokenizing does not perform any model modifications: it merely parses a string into tokens.

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

from abc import ABC
from .plx_scripting_exceptions import PlxScriptingTokenizerError

KEY_POSITION = 'position'
KEY_TYPE = 'type'
KEY_TOKENS = 'tokens'
KEY_ERROR_POSITION = 'errorpos'


class TokenBase(ABC):
    """
    Represents a token
    Properties:
    - type (indicates what type of token it is)
    - value (the parsed representation of the token)
    - position (indicates the start position of the token in the original string)
    - end_position (indicates the end position of the token in the original string)
    - length (the number of characters the token consumed from the original string)
    """
    def __init__(self, raw_data):
        self._raw_data = raw_data
        for key in raw_data:
            value = raw_data[key]

            # Position return from the HTTP REST API is 1-based so we need to convert it to better consistency with
            # python where position of a character in a string is 0-based
            if key == KEY_POSITION:
                value -= 1
            setattr(self, key, value)

    def __repr__(self):
        return '{}.{}({})'.format(self.__module__, self.__class__.__name__, self._raw_data)

    def __str__(self):
        return str(self.value)

    @property
    def end_position(self):
        return self.position + self.length - 1


class TokenIdentifier(TokenBase):
    """Something that will act either as command or as object identifier"""
    pass


class TokenComment(TokenBase):
    """
    A piece of comment, i.e. a sequence of characters starting with # up to the end of the string.
    The value of the token includes the starting # sign.
    Additional properties:
    - content: the text after the # sign (e.g. running in the case of  #running)
    """
    pass


class TokenExternalInterpreter(TokenBase):
    """
    A line that should be executed by an external interpreter
    e.g. /output echo Points.
    The value of the token includes the starting / sign
    Additional properties:
    - interpretername: the name of the interpreter (e.g. output in the case of /output echo Points)
    - externalcommand: the command to be executed (e.g. echo Points)
    - content: (e.g. output echo Points)
    """
    pass


class TokenText(TokenBase):
    """
    Identifies a string, which may be enclosed between 1 or 3 sets of single or double quotes
    The value of a text token includes the surrounding quotes
    Additional properties:
    - content: text inside the quotation marks (e.g. input  in the case of "input")
    """
    pass


class TokenInteger(TokenBase):
    """Identifies a number that can be represented by a 32-bit signed integer"""
    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.value = int(self.value)


class TokenFloat(TokenBase):
    """Identifies a number that can be represented as a floating point value"""
    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.value = float(self.value)


class TokenBracket(TokenBase):
    """
    Identifies a bracket type
    Additional properties:
    - brackettype: can be round, square, curly for (), [] respectively {}
    - bracketstate:  can be open or close for {[( respectively )]}
    """
    pass


class TokenMember(TokenBase):
    """Identifies a bracket type"""
    pass


class TokenOperand(TokenBase):
    """Identifies an operand type"""
    pass


class TokenPlus(TokenBase):
    """Identifies the plus operand type"""
    pass


class TokenMinus(TokenBase):
    """Identifies the minus operand type"""
    pass


class TokenMultiplier(TokenBase):
    """Identifies the multiplier operand type"""
    pass


class TokenDivider(TokenBase):
    """Identifies the divider operand type"""
    pass


class TokenComma(TokenBase):
    """Identifies the comma operand type"""
    pass


class TokenAssign(TokenBase):
    """Identifies the assign operand type"""
    pass


def token_factory(token_raw_data):
    """
    Builds the token object based on the data sent from the HTTP REST API
    :param dict token_raw_data: The original token dictionary send from the HTTP REST API
    :return TokenBase: The token object
    """
    type_to_class_name_mapping = {
        'externalinterpreter': 'ExternalInterpreter',
    }
    token_type = token_raw_data.get(KEY_TYPE)
    class_name = type_to_class_name_mapping.get(token_type)
    if not class_name:
        class_name = token_type.title()
    token_class = globals()['Token{}'.format(class_name)]
    return token_class(token_raw_data)


class TokenizerResultHandler(object):
    """
    Helper class which parses the output of the Plaxis server tokenizer resource into a python object
    """

    def __init__(self, response):
        self.partial_tokens = []
        for key in response:
            if key != KEY_TOKENS:
                attribute_name = key if key != KEY_ERROR_POSITION else 'error_position'
                setattr(self, attribute_name, response[key])
            else:
                for token in response.get(KEY_TOKENS):
                    self.partial_tokens.append(token_factory(token))
        if not self.success:
            self.error_position -= 1
            self.error = 'Unrecognized token at position {}'.format(self.error_position)

    @property
    def tokens(self):
        if self.success:
            return self.partial_tokens
        raise PlxScriptingTokenizerError(self.error)
