"""
Purpose: Allow to define the behavior when an error happens after calling PLAXIS commands

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

from .const import RAISE, INTERPRETER, RETRY, NOCLEAR, PRECONDITION

VALID_BEHAVIOURS = (RAISE, INTERPRETER)
VALID_MODIFIERS = (RETRY, NOCLEAR, PRECONDITION)


class ErrorMode(object):
    """
    Class used to manage the triggered behaviour that happens when an internal server error occurs after making a
    request to Plaxis Remote Scripting Server
    """
    def __init__(self, *args):
        self._behaviour = None
        self._modifiers = set()
        self.start_interpreter_method = None
        self._setup_behavior_and_modifiers_from_init_args(args)

    def __str__(self):
        return "'{}', {}".format(self.behaviour, self.modifiers)

    def _setup_behavior_and_modifiers_from_init_args(self, args):
        for arg in args:
            if arg in VALID_BEHAVIOURS and self._behaviour != INTERPRETER:
                self._behaviour = arg
            if arg in VALID_MODIFIERS:
                self._modifiers.add(arg)

    @property
    def behaviour(self):
        if not self._behaviour:
            return RAISE
        return self._behaviour

    @behaviour.setter
    def behaviour(self, value):
        if value in VALID_BEHAVIOURS:
            self._behaviour = value

    @property
    def modifiers(self):
        return None if len(self._modifiers) == 0 else tuple(sorted(self._modifiers))

    @modifiers.setter
    def modifiers(self, value):
        self._modifiers = set()
        if isinstance(value, tuple):
            for item in value:
                if item in VALID_MODIFIERS:
                    self._modifiers.add(item)
        elif isinstance(value, str) and value in VALID_MODIFIERS:
            self._modifiers.add(value)

    @property
    def should_open_interpreter(self):
        return self.behaviour == INTERPRETER and self.start_interpreter_method

    @property
    def should_retry(self):
        return self.modifiers is not None and RETRY in self.modifiers

    @property
    def should_raise(self):
        return self.behaviour == RAISE

    @property
    def have_precondition(self):
        return self.modifiers and PRECONDITION in self.modifiers

    @property
    def should_clear(self):
        return self.modifiers is None or NOCLEAR not in self.modifiers
