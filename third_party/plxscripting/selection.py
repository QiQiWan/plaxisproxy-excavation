"""
Purpose: provides an object that is used to manipulate the selection
    in PLAXIS.

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

from .const import SELECTION_GET, SELECTION_SET, SELECTION_APPEND, SELECTION_REMOVE


class Selection(object):
    def __init__(self, server):
        self.server = server
        self._objects = []

    def refresh(self):
        self._objects = self.server.call_selection_command(SELECTION_GET)

    def clear(self):
        self.set([])

    def set(self, objects):
        self._objects = self.server.call_selection_command(SELECTION_SET, *objects)

    def append(self, *objects):
        self._objects = self.server.call_selection_command(SELECTION_APPEND, *objects)

    def extend(self, objects):
        self.append(*objects)

    def remove(self, *objects):
        self._objects = self.server.call_selection_command(SELECTION_REMOVE, *objects)

    def pop(self):
        if len(self._objects) > 0:
            self.remove(self._objects[-1])

    def __add__(self, objects):
        if not isinstance(objects, (list, tuple)):
            objects = [objects]

        self.append(*objects)
        return self

    def __sub__(self, objects):
        if not isinstance(objects, (list, tuple)):
            objects = [objects]

        self.remove(*objects)
        return self

    def __len__(self):
        return len(self._objects)

    def __iter__(self):
        return iter(self._objects)

    def __contains__(self, obj):
        return obj in self._objects

    def __repr__(self):
        return repr(self._objects)

    def __getitem__(self, index):
        return self._objects[index]

    def __setitem__(self, index, obj):
        self._objects[index] = obj
        self.set(self._objects)

    def __delitem__(self, index):
        self.remove(self._objects[index])

