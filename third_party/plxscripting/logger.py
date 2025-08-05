"""
Purpose: Interfaces for logging things in the scripting layer.

Subversion data:
    $Id: logger.py 16029 2014-03-28 16:00:15Z tj $
    $URL: https://tools.plaxis.com/svn/sharelib/trunk/PlxObjectLayer/Server/plxscripting/logger.py $

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

import os
import time
import datetime
from json import dumps


def _getDefaultLogPath():
    temp_path = os.path.expandvars("%temp%")
    log_folder_path = os.path.join(temp_path, "PlaxisScriptLogs")
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)

    ms_since_epoch = int(1000 * time.time())
    file_name = "requests_{}.log".format(ms_since_epoch)
    return os.path.join(log_folder_path, file_name)


class Logger(object):
    """A simple logger to keep track of the requests being made."""

    def __init__(self, **kwargs):
        super(Logger, self).__init__()

        if 'file' in kwargs:
            self._file = kwargs['file']
        elif 'path' in kwargs:
            self._file = open(kwargs['path'], 'w')
        else:
            self._file = open(_getDefaultLogPath(), 'w')

        self._request_start_time = 0
        self._payload = ""

    def log_request_start(self, payload):
        self._request_start_time = time.perf_counter()
        self._payload = payload

    def log_request_end(self, request):
        duration = time.perf_counter() - self._request_start_time
        ms_duration = int(1000.0 * duration)
        isodate = datetime.datetime.now().isoformat()
        request_text = "[{date}][{duration}ms][{status}({reason})][{url}]\n".format(
            date=isodate, duration=ms_duration, status=request.status_code,
            reason=request.reason, url=request.url)
        self._file.write(request_text)

        self._file.write("Payload: {}\n".format(self._payload))
        self._file.write("Returned Headers: {}\n".format(dumps(dict(request.headers))))

        # The returned text contains double newlines, which is somewhat annoying to read.
        returned = request.text.replace('\x0D\x0A', '\n')
        self._file.write("Returned: {}\n".format(returned))
        self._file.flush()
