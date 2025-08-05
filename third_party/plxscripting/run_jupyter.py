"""
Purpose: Intended for quickly starting Jupyter notebook
Usages:
    python run_jupyter.py --AppServerAddress=localhost --AppServerPort=10 --AppServerPassword=123
    python run_jupyter.py --AppServerAddress=localhost --AppServerPort=10 --AppServerPassword=123
                          --notebook="C:\folder\my.ipynb"
    python run_jupyter.py --AppServerAddress=localhost --AppServerPort=10 --AppServerPassword=123
                          --notebookdir="C:\"

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
import os
import shutil
from argparse import ArgumentTypeError
from plxscripting.interactive import create_parser, set_paths_of_activated_conda_environment

try:
    import notebook.notebookapp
except Exception as e:
    # The most likely cause for the import to have failed is due to the lack of
    # some paths in the "PATH" environment variable. This is currently
    # happening with a Python 3.7 environment
    print(e)
    set_paths_of_activated_conda_environment()
    import notebook.notebookapp

from plxscripting.plaxis_jupyter_kernel import (ENV_VAR_PLAXIS_SERVER_PASSWORD, ENV_VAR_PLAXIS_SERVER_APP_TYPE,
                                                ENV_VAR_PLAXIS_SERVER_ADDRESS, ENV_VAR_PLAXIS_SERVER_PORT)
from plxscripting.easy import new_server
from plxscripting.const import PLAXIS_2D, PLAXIS_3D, INPUT, OUTPUT, SOILTEST

NOTEBOOK_TEMPLATE_PATHS = {PLAXIS_2D: {INPUT: "template_2d_input.ipynb", OUTPUT: "template_2d_output.ipynb",
                                       SOILTEST: "template_2d_soiltest.ipynb"},
                           PLAXIS_3D: {INPUT: "template_3d_input.ipynb", OUTPUT: "template_3d_output.ipynb",
                                       SOILTEST: "template_3d_soiltest.ipynb"}}
EMPTY_NOTEBOOK_PATH = "empty_notebook.ipynb"


def readable(path, is_dir=False):
    path = path.lower()
    if is_dir and not os.path.isdir(path):
        raise ArgumentTypeError("{} is not a valid directory".format(path))
    elif not is_dir and not os.path.isfile(path):
        raise ArgumentTypeError("{} is not a valid file".format(path))
    if os.access(path, os.R_OK):
        return path
    else:
        raise ArgumentTypeError("{0} is not readable".format(path))


def readable_dir(arg_dir):
    return readable(arg_dir, True)


def is_file(arg_file):
    arg_file = arg_file.lower()
    dirname = os.path.dirname(arg_file)
    readable_dir(dirname)
    basename = os.path.basename(arg_file)
    if not basename.endswith('.ipynb'):
        raise ArgumentTypeError("{0} is not a valid notebook file".format(arg_file))
    return arg_file


def create_notebook(path, is_plaxis_2d, appservertype, create_empty_notebook=False):
    template_notebook_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "template_notebooks")
    if create_empty_notebook:
        final_path = EMPTY_NOTEBOOK_PATH
    else:
        final_path = NOTEBOOK_TEMPLATE_PATHS[PLAXIS_2D if is_plaxis_2d else PLAXIS_3D][appservertype]
    template_notebook_path = os.path.join(template_notebook_path, final_path)
    shutil.copy(template_notebook_path, path)


def set_environment_variables_with_plaxis_vars(server_address, server_port, server_apptype, server_password):
    os.environ[ENV_VAR_PLAXIS_SERVER_ADDRESS] = str(server_address)
    os.environ[ENV_VAR_PLAXIS_SERVER_PORT] = str(server_port)
    os.environ[ENV_VAR_PLAXIS_SERVER_APP_TYPE] = str(server_apptype)
    os.environ[ENV_VAR_PLAXIS_SERVER_PASSWORD] = str(server_password)


if __name__ == '__main__':
    parser = create_parser()
    parser.add_argument('--notebookdir', type=readable_dir,
                        help='The directory where the Jupyter notebooks are located.')
    parser.add_argument('--notebook', type=is_file, help='The notebook to open or create.')
    args = parser.parse_args()
    set_environment_variables_with_plaxis_vars(args.AppServerAddress, args.AppServerPort, args.AppServerType,
                                               args.AppServerPassword)
    s, s.plx_global = new_server(address=args.AppServerAddress, port=args.AppServerPort,
                                 password=args.AppServerPassword)
    notebook_args = []
    if args.notebookdir:
        notebook_args.append('--notebook-dir={}'.format(args.notebookdir))
    if args.notebook:
        if not os.path.isfile(args.notebook):
            create_notebook(args.notebook, s.is_2d, args.AppServerType)
        notebook_args.append('{}'.format(args.notebook))
    sys.exit(notebook.notebookapp.main(notebook_args))

