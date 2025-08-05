"""
Purpose: Gives some very easy-to-use wrappers than can be imported in one go

Subversion data:
    $Id: easy.py 19675 2015-07-01 14:31:55Z tj $
    $URL: https://tools.plaxis.com/svn/sharelib/trunk/PlxObjectLayer/Server/plxscripting/easy.py $

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
import sys

from .const import LOCAL_HOST, ARG_APP_SERVER_ADDRESS, ARG_APP_SERVER_PORT, ARG_PASSWORD, PLAXIS_3D, PLAXIS_2D
from .server import new_server as n_serv
from .console import inplace_console as console
from .console import get_equivalent, ge


def new_server(address=None, port=None, timeout=5.0, request_timeout=None, password=None, error_mode=()):
    invoking_module_namespace = sys._getframe(1).f_locals
    ns_keys = list(invoking_module_namespace.keys())

    if address is None and ARG_APP_SERVER_ADDRESS in ns_keys:
        address = invoking_module_namespace[ARG_APP_SERVER_ADDRESS]

    if port is None and ARG_APP_SERVER_PORT in ns_keys:
        port = invoking_module_namespace[ARG_APP_SERVER_PORT]

    if password is None and ARG_PASSWORD in ns_keys:
        password = invoking_module_namespace[ARG_PASSWORD]

    return n_serv(address=address, port=port, timeout=timeout, request_timeout=request_timeout,
                  password=password, error_mode=error_mode)
