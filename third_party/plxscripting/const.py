"""
Purpose: Constants for the plxscripting package

Subversion data:
    $Id: const.py 19744 2015-07-10 12:52:29Z tj $
    $URL: https://tools.plaxis.com/svn/sharelib/trunk/PlxObjectLayer/Server/plxscripting/const.py $

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

# REST API Authority
LOCAL_HOST = "localhost"
DEFAULT_PORT = 8001

# REST API Resource names
ENVIRONMENT = "environment"
MEMBERS = "members"
NAMED_OBJECTS = "namedobjects"
PROPERTY_VALUES = "propertyvalues"
LIST = "list"
ENUMERATION = "enumeration"
SELECTION = "selection"
EXCEPTIONS = "exceptions"
TOKENIZER = "tokenizer"

# Plaxis commands
PLX_CMD_NEW = "new"
PLX_CMD_CLOSE = "close"
PLX_CMD_RECOVER = "recover"
PLX_CMD_OPEN = "open"

# REST API request strings
ACTION = "action"
COMMANDS = "commands"
NAME = "name"
FILENAME = "filename"
LIST_QUERIES = "listqueries"
GUID = "guid"
TOKENIZE = 'tokenize'

COUNT = "count"
METHOD = "method"
SUBLIST = "sublist"
MEMBERSUBLIST = "membersublist"
INDEX = "index"
MEMBERNAMES = "membernames"
MEMBERINDEX = "memberindex"
STARTINDEX = "startindex"
STOPINDEX = "stopindex"

OWNER = "owner"
PROPERTYNAME = "propertyname"
PHASEGUID = "phaseguid"
STAGED_PREFIX = "staged."
OBJECTS = "objects"

GETLAST = 'getlast'
PEEKLAST = 'peeklast'

# REST API response strings
JSON_COMMANDS = "commands"
JSON_FEEDBACK = "feedback"
JSON_SUCCESS = "success"
JSON_EXTRAINFO = "extrainfo"
JSON_GUID = "guid"
JSON_RETURNED_OBJECTS = "returnedobjects"
JSON_RETURNED_OBJECT = "returnedobject"
JSON_RETURNED_VALUES = "returnedvalues"
JSON_SELECTION = "selection"
JSON_TYPE = "type"
JSON_VALUE = "value"
JSON_QUERIES = "queries"
JSON_CMDLINE_NAME = "commandlinename"
JSON_PROPERTIES = "properties"
JSON_NAMEDOBJECTS = "namedobjects"
JSON_ISLISTABLE = "islistable"
JSON_OWNERGUID = 'ownerguid'
JSON_LISTQUERIES = "listqueries"
JSON_METHODNAME = "methodname"
JSON_OUTPUTDATA = "outputdata"
JSON_MEMBERNAMES = "membernames"
JSON_ENUMVALUES = "enumvalues"
JSON_TYPE_JSON = "JSON"
JSON_KEY_JSON = 'json'
JSON_KEY_CONTENT_TYPE = 'ContentType'
JSON_KEY_REPLY_CODE = 'ReplyCode'
JSON_KEY_CODE = 'Code'
JSON_KEY_RESPONSE = 'Response'
JSON_KEY_REQUEST_DATA = 'RequestData'
JSON_NAME = 'Name'
JSON_HEADERS = 'headers'

# Unit test "recorder"
RECORDER_FOLDER = "recorder"
RECORDER_RESULTS_FOLDER = "recorder_results"
RECORDER_INPUT_FOLDER = "recorder_input"
RECORDER_RESULTS_FILENAME_EXT = ".json"

# Other constants
PLX_GLOBAL = "GLOBAL"
NULL_GUID = '{00000000-0000-0000-0000-000000000000}'

# Command line argument constants
ARG_APP_SERVER_ADDRESS = 'AppServerAddress'
ARG_APP_SERVER_PORT = 'AppServerPort'
ARG_PASSWORD = 'AppServerPassword'
ARG_APP_SERVER_TYPE = 'AppServerType'

# PLAXIS application types to server & global variable suffix
INPUT = "input"
OUTPUT = "output"
SOILTEST = "soiltest"

APP_SERVER_TYPE_TO_VARIABLE_SUFFIX = {
    INPUT: 'i',
    OUTPUT: 'o',
    SOILTEST: 't'
}
APP_SERVER_TYPE_TO_DEFAULT_SERVER_PORT = {
    INPUT: 10000,
    OUTPUT: 10001,
    SOILTEST: 10002
}

# Selection commands
SELECTION_GET = 'get'
SELECTION_SET = 'set'
SELECTION_APPEND = 'append'
SELECTION_REMOVE = 'remove'

# Server names
PLAXIS_2D = 'PLAXIS 2D'
PLAXIS_3D = 'PLAXIS 3D'

PLAXIS_PATH = "plaxis_path"
PLAXIS_VERSION = "plaxis_version"
PLAXIS_BASE_REGEDIT_PATH = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\"

PLAXIS_2D_INPUT_EXECUTABLE_FILENAME = "Plaxis2DXInput.exe"
PLAXIS_CLASSIC_2D_INPUT_EXECUTABLE_FILENAME = "Plaxis2DInput.exe"
PLAXIS_3D_INPUT_EXECUTABLE_FILENAME = "Plaxis3DInput.exe"
PLAXIS_2D_OUTPUT_EXECUTABLE_FILENAME = "Plaxis2DOutput.exe"
PLAXIS_3D_OUTPUT_EXECUTABLE_FILENAME = "Plaxis3DOutput.exe"

# Error mode
INTERPRETER = 'interpreter'
RAISE = 'raise'
RETRY = 'retry'
PRECONDITION = 'precondition'
NOCLEAR = 'noclear'
NUMBER_OF_RETRIES = 1
SECONDS_DELAY_BEFORE_RETRY = 0.2

# HTTP Status codes
INTERNAL_SERVER_ERROR = 500
