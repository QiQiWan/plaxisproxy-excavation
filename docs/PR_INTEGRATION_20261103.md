# 历史 PR #1–#4 整合说明

本分支以 main 的 `8a30dfe2604fab16343e2604b9d667ba9b905786` 为基线，完整整合四个已关闭、未合并的 PR。
发布新的单一 Draft PR 供维护者审批；本次不合并、不发布 PyPI 版本、不启动 PLAXIS 求解。

## 修改来源

| 原 PR | 保留内容 | 整合后的入口 |
| --- | --- | --- |
| #1 | 逐阶段收敛判断、结果导出前验证、可选 IFC 模块导入 | builder/excavation_builder.py、plaxishelper/plaxisrunner.py、包根导入及阶段状态回归 |
| #2 | 只读 V22 API 探测、客户端路径和阶段状态记录 | scripts/probe_plaxis_v22.py、PLAXIS_3D_V22_COMPATIBILITY.md、探测回归 |
| #3 | 分离版本客户端依赖、清理明文默认密码、修正构建元数据 | pyproject.toml、requirements.txt、三份配置、示例、双语 README |
| #4 | 阶段属性快照、联合回读、补偿恢复、保护阶段、实际轴力独立读取 | phase_properties.py、phase_results.py、V22 最小验证脚本及全部恢复回归 |

四个 PR 的准确来源 SHA、当前测试结果和未通过测试清单记录于 `PR_INTEGRATION_20261103_validation.json`。
四份旧分支专用 CI 已由统一的 `.github/workflows/integration.yml` 接替，验证内容均保留，新增安装包联通检查。

## 整合时修复的问题

1. **可选依赖真正延后加载。** 默认安装不含版本专属 plxscripting，原 Runner/Output 的导入链却立即依赖客户端。
   现在纯模型、Builder、阶段属性和结果接口均可在无客户端环境导入；明确请求连接时才加载安装版本的 easy API。
   不使用仓库旧 third_party 作为兼容降级路径。缺少客户端时给出安装提示；已安装客户端的初始化错误原样传递。
2. **空阶段计划先验证再计算。** 无阶段或所有阶段明确禁用时直接报错，不调用 calculate。
3. **公共接口可直接使用。** 阶段属性、预应力和 Output 轴力接口从 plaxishelper 统一导出，测试不再只依赖单文件加载或 AST 提取。
4. **构建后的文件可用。** 修正 MANIFEST 中 README 大小写和过时条目；sdist 包含验证脚本、文档和测试，wheel 包含新增模块。
5. **示例配置一致。** 额外清理遗留测试演示和 Notebook 的明文密码赋值，改从环境变量读取，Notebook 输出和其他内容保留。
6. **统一持续集成。** Python 3.9/3.11 使用 Ubuntu 22.04，执行完整整合契约、脚本编译、wheel/sdist 构建及仓库外安装导入。

## 验证结果与边界

| 项目 | 结果 |
| --- | --- |
| Python 3.9.25 整合回归 | 76 项通过 |
| Python 3.11.17 整合回归 | 76 项通过，另有 35 个子用例通过 |
| 两个版本 wheel / sdist 构建 | 通过 |
| 无 PLAXIS 客户端、仓库外导入已安装 wheel | 两个版本通过 |
| 独立代码审查 | 已完成，客户端初始化异常被吞掉的问题已用失败回归复现并修复 |
| Windows 授权 PLAXIS V22 | **未执行** |
| 全仓库 pytest | **未通过**：8 处收集错误，不能据相关回归通过宣称全套通过 |
| 对 tests 继续收集后运行 | 127 通过、48 失败、4 处收集错误、35 个子用例通过 |

未通过的旧测试主要涉及已变更/缺失的旧 API：plaxismapper、plaxisexcavation、StageSettings、LaterialResistanceType，
以及旧测试对材料、结构、几何和 Runner 接口/字符串的期待。第三方随仓库保存的旧客户端测试还存在 Python 3.11 的 collections.MutableMapping 导入问题。
本次没有通过删除旧测试或新增 xfail 隐去这些结果；统一 CI 另行执行标记为非阻断的旧测试审计，并上传日志。
本次通过的 76 项只证明四个 PR 的整合接口与新增安装路径契约，不包含真实求解器或所有历史功能。

## 仍需维护者确认

- V22 本机客户端、实际 calculate 命令的成功返回形式和逐阶段 CalculationResult 读取。
  当前计算命令适配器仍保留 #1 的严格 `OK` 判定，其他成功标记需根据真实版本证据适配。
- 真实阶段继承与显式覆盖、压缩符号、支撑激活和连接、Output 结果字段及力平衡。
- 阶段补偿只验证标量快照，不能保证恢复 PLAXIS 内部继承标记，无法约束 GUI 或其他进程写入。
- 旧测试/API 迁移需后续单独处理，当前分支不能作为全项目已验证可运行的声明。

配置与只读探测见 `PLAXIS_3D_V22_COMPATIBILITY.md`；阶段属性与实际轴力的试验副本操作见 `phase-prestress-v22-validation.md`。
`phase-prestress-validation-2026-10-27.json` 保留为 #4 的历史记录，当前整合状态以本页和 20261103 验证记录为准。
