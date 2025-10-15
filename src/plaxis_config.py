# config/plaxis_config.py
# -*- coding: utf-8 -*-
"""
Plaxis 3D 远程连接配置
- 默认按本机配置运行；也可通过环境变量覆盖，便于CI/多环境切换。
  环境变量：
    PLAXIS_HOST               (默认: "localhost")
    PLAXIS_PORT               (默认: "10000")
    PLAXIS_PASSWORD           (默认: 见下方)
    PLAXIS_DIR                (默认: 见下方安装目录)
    PLAXIS_PATH               (默认: <PLAXIS_DIR>/Plaxis3DInput.exe)
    PLAXIS_START              (默认: "1"，启动新进程；"0" 表示仅连接)
    PLAXIS_STARTUP_TIMEOUT_S  (默认: "8.0")
"""

import os
from pathlib import Path

# 基本连接信息
HOST = os.getenv("PLAXIS_HOST", "localhost")
PORT = int(os.getenv("PLAXIS_PORT", "10000"))
PASSWORD = os.getenv("PLAXIS_PASSWORD", "yS9f$TMP?$uQ@rW3")

# 安装路径 / 可执行文件（可被环境变量覆盖）
DEFAULT_PLAXIS_DIR = os.getenv(
    "PLAXIS_DIR",
    r"D:\Program Files\Bentley\Geotechnical\PLAXIS 3D CONNECT Edition V22"
)
PLAXIS_PATH = os.getenv(
    "PLAXIS_PATH",
    str(Path(DEFAULT_PLAXIS_DIR) / "Plaxis3DInput.exe")
)

# 启动/连接行为
# True：由测试/工具类启动一个新的 Plaxis Input 进程；False：仅连接到已启动的服务
START_NEW_PROCESS = os.getenv("PLAXIS_START", "1").lower() not in ("0", "false")
STARTUP_TIMEOUT_S = float(os.getenv("PLAXIS_STARTUP_TIMEOUT_S", "8.0"))
