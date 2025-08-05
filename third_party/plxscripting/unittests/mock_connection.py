"""
Purpose: Mock connection.py module for the unittests

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

MOCK_GUID_FORMAT = '{}{:012x}{}'
MOCK_GUID_PREFIX = '{FFFFFFFF - FFFF - FFFF - FFFF - '
MOCK_GUID_SUFFIX = '}'

MOCK_EXCEPTION = '''operating system   : MockOS 1.0 build 7
program up time    : 13 minutes 37 seconds
processors         : 1x Plaxis UnitTestCore© CPU @ 0.0GHZ
physical memory    : 0/0 MB (free/total)
free disk space    : (C:) 0 GB (D:) 0 GB
executable         : PlaxisXDXput.exe
version            : 20xx.x.0.0
exception class    : Exception
exception message  : Mock exception 

thread $0000: <priority:-1>
ffffffff +ff PlaxisXDXput.exe                 1697 +1 TPTProcedureRunner.Execute
ffffffff +ff PlaxisXDXput.exe PlxTasks         701 +7 TPlxTask.ExecuteWrapped
ffffffff +ff PlaxisXDXput.exe PlxTasks         531 +9 ThreadProc
ffffffff +ff PlaxisXDXput.exe System                  ThreadWrapper
ffffffff +ff KERNEL32.DLL                             BaseThreadInitThunk
ffffffff +ff ntdll.dll                                RtlUserThreadStart'''


class HTTPConnection:
    MAGIC_REQUEST_OBJECT = 'give_me_an_object'

    def __init__(self, host, port, timeout=5.0, request_timeout=None, password='', error_mode=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.request_timeout = request_timeout
        self._password = password
        self.error_mode = error_mode
        self.logger = None

        self.test_poll_status = True
        self.test_success = True
        self.test_exception_cleared = False
        self.test_tokenize_external = False
        self._guid_counter = -1
        self._selection = []

    @property
    def _next_guid(self):
        self._guid_counter += 1
        return MOCK_GUID_FORMAT.format(MOCK_GUID_PREFIX, self._guid_counter, MOCK_GUID_SUFFIX).upper()

    def poll_connection(self):
        return self.test_poll_status

    def request_environment(self, command_string, filename=''):
        if self.test_success:
            return 'OK'
        else:
            return '???'  # environment commands cannot fail gracefully, they crash plaxis -_-"

    def request_commands(self, *commands):
        reply = {'commands': [],
                 'ReplyCode': '0' * 32}

        for command in commands:
            if self.test_success:
                reply['commands'].append({
                    'feedback': {
                        'extrainfo': 'Reply_to_command: {}'.format(command),
                        'returnedobjects': [
                            {
                                'islistable': False,
                                'guid': self._next_guid,
                                'type': 'Mock_reply_object_{}'.format(self._guid_counter)
                            }
                        ] if HTTPConnection.MAGIC_REQUEST_OBJECT in command else [],
                        'debuginfo': '',
                        'success': True,
                        'errorpos': -1,
                        'returnedvalues': []
                    },
                    'command': command
                })
            else:
                reply['commands'].append({
                    'feedback': {
                        'extrainfo': 'Cannot intersect unless there is at least one volume or surface in the geometry',
                        'debuginfo': '',
                        'success': False,
                        'errorpos': -1},
                    'command': command  # 'gotomesh'
                })

        return reply

    def request_members(self, *guids):
        reply = {'queries': {},
                 'ReplyCode': '0' * 32}

        for guid in guids:
            reply['queries'].update({guid: {
                'extrainfo': '',
                'success': True,
                'properties': {
                    'Mock_number': {'islistable': False,
                                    'value': 42,
                                    'type': 'Number',
                                    'guid': self._next_guid,
                                    'ispublished': True,
                                    'ownerguid': guid,
                                    'caption': 'Mock_number#'},
                    'TypeName': {'islistable': False,
                                 'value': 'MockItem',
                                 'type': 'Text',
                                 'guid': self._next_guid,
                                 'ispublished': False,
                                 'ownerguid': guid,
                                 'caption': 'TypeName'},
                    'IsDynamicComponent': {'islistable': False,
                                           'value': False,
                                           'type': 'Boolean',
                                           'guid': self._next_guid,
                                           'ispublished': False,
                                           'ownerguid': guid,
                                           'caption': 'IsDynamicComponent'},
                    'Name': {'islistable': False,
                             'value': 'MockItem_Name_memb',
                             'type': 'Text',
                             'guid': self._next_guid,
                             'ispublished': False,
                             'ownerguid': guid,
                             'caption': 'Name'},
                    'UserFeatures': {'islistable': False,
                                     'value': {'islistable': True,
                                               'type': 'PlxUserFeatureList',
                                               'guid': self._next_guid},
                                     'type': 'Object',
                                     'guid': self._next_guid,
                                     'ispublished': False,
                                     'ownerguid': guid,
                                     'caption': 'UserFeatures'},
                },
                'commands': ['echo'],
                'commandlinename': 'LineLoad_1'}})
        return reply

    def request_namedobjects(self, *object_names):
        reply = {'namedobjects': {},
                 'ReplyCode': '0' * 32}  # 32 chars

        for obj in object_names:
            reply['namedobjects'][obj] = {
                'extrainfo': '',
                'success': True,
                'returnedobject': {
                    'islistable': False,
                    'type': 'MockItem_{}'.format(self._guid_counter),
                    'guid': self._next_guid
                }
            }
        return reply

    def request_propertyvalues(self, owner_guids, property_name, phase_guid=""):
        if phase_guid:
            raise Exception('Phase guid not supported')

        reply = {'queries': [{owner_guid: {}} for owner_guid in owner_guids],
                 'ReplyCode': '0' * 32}  # 32 chars

        if property_name == 'Name':
            for (i, owner_guid) in enumerate(owner_guids):
                reply['queries'][i][owner_guid].update({
                    'extrainfo': '',
                    'success': True,
                    'properties': {'Name': 'MockItem_Name_pval'}
                })

        elif property_name == 'UserFeatures':
            for (i, owner_guid) in enumerate(owner_guids):
                reply['queries'][i][owner_guid].update({
                    'extrainfo': '',
                    'success': True,
                    'properties': {
                        'UserFeatures': {
                            'islistable': False,
                            'type': 'MockItem_{}'.format(self._guid_counter),
                            'guid': self._next_guid
                        }
                    }
                })
        else:
            raise Exception('property_name "{}" not supported'.format(property_name))

        # in case of a single owner query, use the legacy signature
        if len(owner_guids) == 1:
            reply['queries'] = reply['queries'][0]

        return reply

    def request_list(self, *list_queries):
        reply = {'listqueries': [],
                 'ReplyCode': '0' * 32}  # 32 chars

        for query in list_queries:
            if query['method'] == 'count':
                reply['listqueries'].append({
                    'extrainfo': '',
                    'success': True,
                    'methodname': 'count',
                    'guid': query['guid'],
                    'outputdata': 3
                })
            elif query['method'] == 'index':
                reply['listqueries'].append({
                    'extrainfo': '',
                    'success': True,
                    'startindex': query['startindex'],
                    'methodname': 'index',
                    'guid': query['guid'],
                    'outputdata': {
                        'islistable': False,  # Let's not nest lists...
                        'type': 'MockItem_{}'.format(self._guid_counter),
                        'guid': self._next_guid}
                })
            elif query['method'] == 'memberindex':
                reply['listqueries'].append({
                    'extrainfo': '',
                    'success': True,
                    'startindex': query['startindex'],
                    'methodname': 'index',
                    'membernames': query['membernames'],
                    'guid': query['guid'],
                    'outputdata': {}  # will be filled in the for loop below
                })

                for member_name in query['membernames']:
                    reply['listqueries'][-1]['outputdata'][member_name] = {
                        'islistable': False,  # Let's not nest lists...
                        'type': 'MockNumber',
                        'guid': self._next_guid,
                        'ownerguid': self._next_guid,
                    }
            elif query['method'] == 'sublist':
                reply['listqueries'].append({
                    'extrainfo': '',
                    'stopindex': query['stopindex'],
                    'success': True,
                    'startindex': query['startindex'],
                    'methodname': 'sublist',
                    'guid': query['guid'],
                    'outputdata': []  # will be filled in the for loop below
                })

                for x in range(int(query['stopindex']) - int(query['startindex'])):
                    reply['listqueries'][-1]['outputdata'].append({
                        'islistable': False,  # Let's not nest lists...
                        'type': 'MockItem_{}'.format(self._guid_counter),
                        'guid': self._next_guid
                    })
            elif query['method'] == 'membersublist':
                reply['listqueries'].append({
                    'extrainfo': '',
                    'stopindex': query['stopindex'],
                    'success': True,
                    'startindex': query['startindex'],
                    'membernames': query['membernames'],
                    'methodname': 'membersublist',
                    'guid': query['guid'],
                    'outputdata': {}  # will be filled in the for loop below
                })

                for member_name in query['membernames']:
                    reply['listqueries'][-1]['outputdata'][member_name] = []
                    for x in range(int(query['stopindex']) - int(query['startindex'])):
                        reply['listqueries'][-1]['outputdata'][member_name].append({
                            'islistable': False,  # Let's not nest lists...
                            'type': 'MockNumber',
                            'guid': self._next_guid,
                            'ownerguid': self._next_guid,
                        })

            else:
                raise Exception(f"Unsupported method: {query['method']}")

        return reply

    def request_enumeration(self, *guids):
        reply = {'queries': {},
                 'ReplyCode': '0' * 32}

        for guid in guids:
            reply['queries'][guid] = {
                'extrainfo': '',
                'success': True,
                'enumvalues': {
                    'item_0': 0,
                    'item_2': 2,
                    'item_1': 1,
                    'item_3': 3
                }
            }
        return reply

    def request_selection(self, command, *guids):
        reply = {'selection': [],
                 'ReplyCode': '0' * 32}

        if command == 'get':
            pass
        elif command == 'set':
            self._selection = []
            for guid in guids:
                self._selection.append({
                    'islistable': False,
                    'type': 'MockItem_Selection',
                    'guid': guid
                })
        elif command == 'append':
            for guid in guids:
                self._selection.append({
                    'islistable': False,
                    'type': 'MockItem_Selection',
                    'guid': guid
                })
        elif command == 'remove':
            newselection = []
            for selecteditem in self._selection:
                if selecteditem['guid'] not in guids:
                    newselection.append(selecteditem)
            self._selection = newselection
        else:
            raise Exception(f"Invalid command: {command}")

        reply['selection'] = self._selection
        return reply

    def request_server_name(self):
        return 'mock_connection_py'

    def request_exceptions(self, clear=True):
        reply = ''
        if not self.test_exception_cleared:
            reply = MOCK_EXCEPTION

        self.test_exception_cleared = clear

        return reply

    def request_tokenizer(self, commands):
        reply = {'tokenize': [],
                 'ReplyCode': '0' * 32}
        for command in commands:
            if self.test_success:
                if self.test_tokenize_external:
                    reply['tokenize'].append({
                        'extrainfo': '',
                        'tokenize': command,  # '/command object "param" "param2" 2 3 4'
                        'tokens': [{
                            'interpretername': 'command',
                            'position': 1,
                            'externalcommand': 'object "param" "param2" 2 3 4',
                            'length': 38,
                            'value': '/command object "param" "param2" 2 3 4',
                            'content': '/command object "param" "param2" 2 3 4',
                            'type': 'externalinterpreter'}],
                        'success': True,
                        'errorpos': -1
                    })
                else:
                    reply['tokenize'].append({
                        'extrainfo': '',
                        'tokenize': command,  # 'command object "param" "param2" 2 3 4'
                        'tokens': [{'position': 1, 'length': 7, 'value': 'command', 'type': 'identifier'},
                                   {'position': 9, 'length': 6, 'value': 'object', 'type': 'operand'},
                                   {'position': 16, 'length': 7, 'value': '"param"', 'content': 'param',
                                    'type': 'text'},
                                   {'position': 24, 'length': 8, 'value': '"param2"', 'content': 'param2',
                                    'type': 'text'},
                                   {'position': 33, 'length': 1, 'value': '2', 'type': 'integer'},
                                   {'position': 35, 'length': 1, 'value': '3', 'type': 'integer'},
                                   {'position': 37, 'length': 1, 'value': '4', 'type': 'integer'}
                                   ],
                        'success': True,
                        'errorpos': -1
                    })
            else:
                reply['tokenize'].append({
                    'extrainfo': 'Unbalanced quotes',
                    'tokenize': command,  # '1234 "bla bla 542 2'
                    'tokens': [
                        {'position': 1, 'length': 4, 'value': '1234', 'type': 'integer'}
                    ],
                    'success': False,
                    'errorpos': 6
                })

        return reply
