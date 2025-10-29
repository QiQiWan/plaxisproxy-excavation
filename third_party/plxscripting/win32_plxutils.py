"""
A collection of help/utility functions that interact with win32

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
import win32gui
import win32con
import win32process


def get_windows_for_pid(pid):
    def callback(window, windows):
        if win32gui.IsWindowVisible(window) and win32gui.IsWindowEnabled(window):
            _, found_pid = win32process.GetWindowThreadProcessId(window)
            if found_pid == pid:
                windows.append(window)
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows


def hide_windows(pid):
    windows = get_windows_for_pid(pid)
    for window in windows:
        win32gui.ShowWindow(window, win32con.SW_HIDE)
