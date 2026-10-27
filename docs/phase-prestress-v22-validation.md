# PR #4：阶段属性、预应力与实际轴力验证

## 本次修改的边界

2026-10-27：本次交付包含无许可证模拟回归与 Windows PLAXIS 3D V22 最小验证脚本。
真实 V22 **尚未运行**；不得据模拟测试宣称压缩符号、阶段继承、连接关系或实际轴力已验证。
保持 Draft；不合并、不启动正式工程计算。PR #1/#2/#3 的兼容性问题不在此补丁中顺带修改。

## 写入与失败恢复

`set_phase_property()` 现在要求旧的阶段值可读。新增 `set_phase_properties()` 可按映射顺序写入一组属性。
`set_node_to_node_prestress()` 先读取同一目标阶段的 `AdjustPrestress` 和 `PrestressForce`，
验证布尔/数值后才允许第一条写命令。写入仍使用 `g_i.set(prop, phase, value)`，无全局赋值降级路径。

每条写命令都回读，结束后再次联合回读全部属性。若当前值已相同，则跳过写入；重复调用不会重复施加写命令。
这里的幂等性只涉及 Input 配置，不能据此推断重新求解的工程行为。

一旦尝试过写入，异常时按逆序尝试恢复目标阶段的**完整属性快照**，包含可能被首条命令连带改变的属性，以及“服务端已执行、响应丢失”的那一条。
某条恢复失败后继续尝试其余属性，最后再次读取全部目标属性与保护阶段。
这属于补偿恢复，PLAXIS 原生事务、继承标记恢复、GUI 撤销栈和崩溃恢复均未获得保证。
恢复命令本身也可能存在属性耦合，最终联合回读未通过的状态必须人工处理。KeyboardInterrupt/SystemExit 会先触发恢复再原样抛出，异常对象的 phase_property_error 保存恢复证据。

`PhasePropertyError` 提供：

| 字段 | 含义 |
| --- | --- |
| `stage` | `preflight` 或 `write/readback` |
| `before` | 写前读取到的标量快照，键包含 target/guard 编号 |
| `observed` | 恢复后最终回读；无法读取时标注 unreadable |
| `rollback_verified` | `True`：监测范围内标量已恢复；`False`：恢复未验证；`None`：写前失败 |
| `rollback_errors` | 恢复命令或最终回读的问题列表 |
| `recovery_required` | `True` 时必须中止后续批处理、保存和求解，人工检查或重新打开未修改副本 |

即使 `rollback_verified=True`，本次操作仍抛出异常，调用方不得当作写入成功。
恢复命令自身报错而最终值确实恢复时，`rollback_verified` 可以为 True，命令异常仍保留在列表中。
不要吞掉异常后继续跑后续工况，不自动重试未知状态。

`guard_phases=(previous_phase, ...)` 会对保护阶段做写前/写后检查。保护阶段永不自动写入，避免破坏继承关系。
若目标恢复成功但保护阶段变化，仍标记 `recovery_required=True`。
未传入的阶段不在监测范围。建议至少包含前一阶段及已有的直接后续阶段。
真实 PLAXIS 的继承可能自然引起子阶段值变化，应由实际 V22 验证确认；不能把所有变化都简单认定为全局写入。

模块内 RLock 串行化本进程通过该网关进行的操作，无法约束 GUI、其他进程或绕过网关的写入。
使用期间需独占 Input 会话。快照只包含属性标量，恢复相同值可能改变显式/继承状态，正式模型应使用未修改副本。

```python
from plaxisproxy_excavation.plaxishelper.phase_properties import (
    PhasePropertyError, set_node_to_node_prestress,
)
try:
    command = set_node_to_node_prestress(
        g_i, support_i, phase_i, signed_force_kn,
        compression_sign_verified=True,  # 仅在本机实际确认符号后填写
        guard_phases=(previous_phase_i,),
    )
except PhasePropertyError as exc:
    # 记录 before / observed / rollback_errors；终止本次任务。
    # recovery_required=True 时先人工恢复，不能继续求解。
    raise
```

调用方还需确认：模型力单位是 kN、支撑已激活、目标阶段适合预应力操作、材料/连接关系正确。
通用属性网关不猜测对象 Active 字段或阶段类型，也不自动激活支撑。

## 命令值与实际轴力

Input 返回的 `command_prestress_kn` 是每根锚杆/支撑的命令力，`actual_axial_force_kn` 始终为 None。
仓库现有 `PlaxisOutput.get_results()` 面向已绑定阶段，且存在 node/stresspoint 回退。
本补丁新增独立的 `phase_results.read_node_to_node_axial_force()`，明确传入 Output 支撑、Output 阶段和结果类型，
不修改旧接口，不默默回退到别的阶段、结果类型或位置。

```python
from plaxisproxy_excavation.plaxishelper.phase_results import read_node_to_node_axial_force
actual = read_node_to_node_axial_force(
    g_o, support_o, phase_o,
    result_type=g_o.ResultTypes.NodeToNodeAnchor.AnchorForce3D,
    force_unit='kN', location='node',
)
```

结果类型路径需在本机 V22 的 Output 命令帮助中确认；其他版本的 N 或其他名称不自动替换。
接口拒绝文本、空结果、非有限数、布尔值和未经确认的单位。返回 `axial_force_kn` 列表，保留全部有符号样本。
不能把两个端点的同一轴力相加作为支撑反力。结果非空也不能证明阶段收敛、阶段对应正确或模型力学正确；
调用方需先在 Output 确认收敛、阶段/支撑身份、单位、局部符号以及支撑两端连接。

公开参考：[Seequent PLAXIS 3D 2025.1 Reference Manual](https://files.seequent.com/PLAXIS/Manuals/PLAXIS_3D/English/PLAXIS_3D_2_Reference%20Manual.pdf)，
12.1.3 节说明命令力按每根锚杆给定。该手册晚于 V22，不能代替 V22 验证；请优先以安装版本的 Reference Manual 和 Scripting Reference 为准。

## Windows V22 最小验证

准备一个**可丢弃的简化模型副本**：一根已连接、激活的 node-to-node 支撑，两个施工阶段。
目标阶段和保护阶段应具有明确可读的属性；从简单基准先手动确认本机压缩符号、kN 单位。
使用 V22 自带 Python/plxscripting 环境，在 GUI 打开本地 Remote Scripting 服务。下列端口、名称必须改为实际值。
脚本不创建/打开模型、不保存、不激活、不划分网格、不调用求解器。不要连接正在运行的正式工程会话。

PowerShell：

```powershell
# 使用当前进程环境变量传递 Remote Scripting 密码；不写入命令历史或报告。
$secret = Read-Host 'PLAXIS Remote Scripting password' -AsSecureString
$env:PLAXIS_PASSWORD = [System.Net.NetworkCredential]::new('', $secret).Password
# 先只读，核对快照、对象、阶段、版本与单位。
python scripts/verify_v22_phase_prestress.py --mode input --port 10000 --support NodeToNodeAnchor_1_1 --phase Phase_7 --force-unit kN --build 22.1.0.452 --report evidence/read-only.json
```

确认当前是试验副本，并已独立验证压缩符号后执行下列配置写入测试；`-120` 仅演示参数位置，需按本机符号修改：

```powershell
python scripts/verify_v22_phase_prestress.py --mode input --port 10000 --support NodeToNodeAnchor_1_1 --phase Phase_7 --guard-phase Phase_6 --force -120 --force-unit kN --build 22.1.0.452 --write-test --confirm-scratch-model --compression-sign-verified --report evidence/input-normal.json
```

脚本先落盘快照，写入/联合回读/重复调用，然后在 finally 中恢复原配置并检查保护阶段。
要验证故障恢复，在同一命令分别添加 `--failure before` 和 `--failure after`，报告改成独立文件。
两种故障分别在第二条命令执行前、执行后抛异常，走生产补偿路径；快照需满足 AdjustPrestress=False 且旧力不同于测试力。
成功报告应有 `input_contract_passed=True`、`cleanup_verified=True`；故障报告中还需 `expected_failure.rollback_verified=True`。
没有任何成功 JSON 由本次 Linux 模拟运行冒充生成。

随后关闭副本且不保存，以保留原始继承关系。JSON 中的 build/单位/符号是操作者确认信息，并非脚本自动鉴定。
系统断电、进程被杀或远程服务失联时 finally 可能无法完成，依照落盘 before 快照人工检查；不要保存状态不明的副本。

实际力需另行验证：由工程师在简化试验副本中设置并手动求解；至少检查施加预应力阶段，以及后续施工但不再次张拉的阶段。
观察实际轴力如何随施工发展，并检查阶段力平衡、支撑两端连接和符号。允许误差应由该基准力学条件与收敛精度确定，
不要要求所有后续阶段实际力都等于初始命令力。

对**已经收敛并打开 Output 的试验模型**只读提取：

```powershell
python scripts/verify_v22_phase_prestress.py --mode output --port 10001 --support NodeToNodeAnchor_1 --phase Phase_7 --force-unit kN --build 22.1.0.452 --confirm-converged --result-type NodeToNodeAnchor.AnchorForce3D --report evidence/output-phase7.json
Remove-Item Env:PLAXIS_PASSWORD
```

Input/Output 对象名可能不同，不能直接复用索引或代理对象；请在 Output 确认名称和阶段身份。
若本机结果位置要求 stresspoint，显式传入 `--location stresspoint` 并记录原因。
此提取只证明收到数据，脚本不会将其自动判定为力学验证通过。

## 回归与验收状态

- Python 3.9/3.11：执行 `python -m unittest discover -s tests -p 'test_phase_*.py' -v`。
- CI 明确命名为无许可证契约测试，编译 Windows 脚本不代表执行 Windows/V22。
- 覆盖：第二次写入失败、执行后响应丢失、首写失败、恢复失败继续处理、恢复后不可读、缺属性、错误阶段、
  代理属性/值访问异常、非法布尔/数值、联合回读漂移、重复写入、跨阶段保护、Output 数据校验及脚本恢复编排。
- 真实 Windows PLAXIS V22：**NOT RUN**。压缩符号、继承/隔离、收敛、Output 类型与实际轴力/传力平衡仍为 Draft 验收门槛。

本次本地结果：Python 3.9.25/3.11.17 各 41 项模拟测试通过；重点 pytest 41 项及 33 个子用例通过。
全仓库 pytest 在收集阶段出现 11 处错误，原始 PR HEAD 同环境复跑为相同 11 处；详情见同目录 phase-prestress-validation-2026-10-27.json。
因此不能宣称全仓库回归通过。既有导入/第三方兼容问题应作为单独工作项处理。
