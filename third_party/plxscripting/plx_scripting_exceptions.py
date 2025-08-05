"""
Purpose: exceptions for the plxscripting package.

Subversion data:
    $Id: plx_scripting_exceptions.py 13488 2013-09-04 13:21:01Z ac $
    $URL: https://tools.plaxis.com/svn/sharelib/trunk/PlxObjectLayer/Server/plxscripting/plx_scripting_exceptions.py $

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


class PlxScriptingError(Exception):
    """
    Base class for exceptions raised due to misused input/output commands.
        This exception always indicates a server-side problem.
    """


class EncryptionError(Exception):
    """ Can be raised for all kinds of encryption related problems """


class PlxScriptingLocalError(Exception):
    """ This exception can be raised for client-side errors
    that happens before we send commands to server or while
    we processing the results"""


class PlxScriptingPreconditionError(Exception):
    """ Base class for exceptions raised when there was an error on server-side before the request was made """


class PlxScriptingTokenizerError(Exception):
    """ Base class for exceptions raised when there was an error while tokenizing """
