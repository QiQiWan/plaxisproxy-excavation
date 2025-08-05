"""
Purpose: Intended for allowing import of the scripting wrapper automatically on Jupyter notebooks

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
from ipykernel.ipkernel import IPythonKernel
from ipykernel.comm import CommManager
from plxscripting.easy import *
from plxscripting.console import build_instructions, build_splash_lines, get_IPython_module
from plxscripting.const import APP_SERVER_TYPE_TO_VARIABLE_SUFFIX, APP_SERVER_TYPE_TO_DEFAULT_SERVER_PORT, INPUT

ENV_VAR_PLAXIS_SERVER_ADDRESS = 'PLAXIS_SERVER_ADDRESS'
ENV_VAR_PLAXIS_SERVER_PORT = 'PLAXIS_SERVER_PORT'
ENV_VAR_PLAXIS_SERVER_PASSWORD = 'PLAXIS_SERVER_PASSWORD'
ENV_VAR_PLAXIS_SERVER_APP_TYPE = 'PLAXIS_APP_TYPE'


class PlaxisIPythonKernel(IPythonKernel):
    def __init__(self, **kwargs):
        super(IPythonKernel, self).__init__(**kwargs)
        self._init_from_environment()

        # Initialize the InteractiveShell subclass
        self.user_ns = self.get_user_ns()
        self.shell = self.shell_class.instance(parent=self,
            profile_dir = self.profile_dir,
            user_module = self.user_module,
            user_ns     = self.user_ns,
            kernel      = self,
        )
        self.shell.displayhook.session = self.session
        self.shell.displayhook.pub_socket = self.iopub_socket
        self.shell.displayhook.topic = self._topic('execute_result')
        self.shell.display_pub.session = self.session
        self.shell.display_pub.pub_socket = self.iopub_socket

        self.comm_manager = CommManager(parent=self, kernel=self)

        self.shell.configurables.append(self.comm_manager)
        comm_msg_types = ['comm_open', 'comm_msg', 'comm_close']
        for msg_type in comm_msg_types:
            self.shell_handlers[msg_type] = getattr(self.comm_manager, msg_type)

    def _init_from_environment(self):
        self.appservertype = os.environ.get(ENV_VAR_PLAXIS_SERVER_APP_TYPE, INPUT)
        self.address = os.environ.get(ENV_VAR_PLAXIS_SERVER_ADDRESS, 'localhost')
        self.password = os.environ.get(ENV_VAR_PLAXIS_SERVER_PASSWORD, '')
        self.port = os.environ.get(ENV_VAR_PLAXIS_SERVER_PORT, APP_SERVER_TYPE_TO_DEFAULT_SERVER_PORT[self.appservertype])
        self.server_object_name = "s_{}".format(APP_SERVER_TYPE_TO_VARIABLE_SUFFIX[self.appservertype])
        self.global_object_name = "g_{}".format(APP_SERVER_TYPE_TO_VARIABLE_SUFFIX[self.appservertype])

    def get_user_ns(self):
        s, g = new_server(self.address, self.port, password=self.password)
        namespace = {
            self.server_object_name: s,
            self.global_object_name: g,
            'get_equivalent': get_equivalent,
            'ge': ge
        }
        return namespace

    @property
    def banner(self):
        ipython = get_IPython_module()
        splash_lines = build_splash_lines(ipython) + ['']
        banner = '\n'.join(splash_lines)
        banner += build_instructions(self.address, self.port, self.server_object_name, self.global_object_name)
        return banner


def main():
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=PlaxisIPythonKernel)


if __name__ == '__main__':
    main()
