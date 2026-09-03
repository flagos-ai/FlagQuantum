# FlagQuantum 多会话并行开发手册

## 目标

多个开发会话可以从同一架构基线并行工作，但不得共享 Git 工作目录、暂存区或未提交
状态。`FlagQuantum-vNext` 是集成工作区；各团队只在自己的 linked worktree 和分支中
开发。

团队、分支、工作区和路径所有权的机器可读来源是仓库根目录的
`team-ownership.toml`，本文不建立第二套名单。

## 工作区模型

```text
codex/flagquantum-vnext-architecture   FlagQuantum-vNext             集成
codex/vnext-team-core                 FlagQuantum-vNext-core        Core
codex/vnext-team-compiler             FlagQuantum-vNext-compiler    Compiler
codex/vnext-team-runtime              FlagQuantum-vNext-runtime     Runtime
codex/vnext-team-simulation           FlagQuantum-vNext-simulation  Simulation
codex/vnext-team-platform-providers   FlagQuantum-vNext-platform    Platform Provider
codex/vnext-team-execution-providers  FlagQuantum-vNext-execution   Execution Provider
codex/vnext-team-ecosystem            FlagQuantum-vNext-ecosystem   Ecosystem
codex/vnext-team-agent-services       FlagQuantum-vNext-agent       Agent Services
```

每个 Codex 会话只打开其中一个目录。团队工作区内不得切换到其他团队分支，也不得将其他
工作区的未提交文件复制进来。

## 会话启动检查

每个会话开始工作前必须：

1. 阅读根 `AGENTS.md` 和所修改目录中最近的 `AGENTS.md`；
2. 确认当前分支与 `team-ownership.toml` 中该团队的分支一致；
3. 确认工作区干净；
4. 阅读总体架构、相关专题设计、能力成熟度和 Stable Core 保护政策；
5. 明确本次工作的输入契约、输出契约、失败语义和验收测试；
6. 在修改前运行团队范围预检。

显式文件预检示例：

```bash
python tools/check_team_scope.py \
  --team compiler \
  --files flagquantum/_compiler/pipeline.py tests/unit/test_pipeline.py
```

分支完成检查示例：

```bash
python tools/check_team_scope.py \
  --team compiler \
  --base codex/flagquantum-vnext-architecture
```

在系统 Git 不可用的环境中，可通过 `FLAGQUANTUM_GIT` 指定 Git 可执行文件。

## 修改权限

### 团队可直接修改

- `team-ownership.toml` 中本团队最具体的 `owns` 路径；
- 相关测试、示例、基准和非受保护文档；
- 不改变跨领域契约的模块内部实现。

### 必须由集成工作区协调

- Stable Core、公共 API 和序列化模式；
- `contracts/`、`architecture.toml`、`team-ownership.toml`、根 `AGENTS.md`；
- 总体架构和 ADR；
- 依赖清单、CI 和发布配置；
- 新跨领域契约、Provider 类型或长期兼容层。

团队遇到受保护区域时应提交契约/ADR 提案，不得在本团队分支复制一个私有替代类型。

## 路径所有权判定

路径采用“最具体规则优先”：

- `runtime/**` 默认属于 Runtime；
- `runtime/backends/**` 过渡期属于 Simulation；
- `runtime/platforms/**` 过渡期属于 Platform；
- `runtime/target_execution.py` 过渡期属于 Execution Provider；
- `extensions/**` 默认属于 Ecosystem；
- `extensions/sdk/**` 过渡期属于 Platform。

这使当前代码在尚未移动目录时也只有一个责任团队。迁移到目标目录后，应同步删除旧规则。

## 契约优先的集成顺序

跨团队功能分两次集成：

```text
契约/ADR 提案
 -> 集成工作区批准并合入契约、假实现和契约测试
 -> 团队分支同步新基线
 -> 各团队独立实现
 -> 团队范围检查与模块测试
 -> 集成工作区合并
 -> 跨实现一致性和替换测试
```

不得在一个巨大变更中同时修改 Core 契约、Compiler、Runtime、Simulation 和 Provider。

## 同步与合并

- 团队分支通过合并集成分支获得最新契约；不得在其他团队工作区直接改代码；
- 一个提交尽量只包含一个团队范围；跨团队变更拆成有顺序的提交；
- 合并前运行团队范围检查、架构检查、相关单元测试和差分/替换测试；
- 集成负责人检查能力声明、回退证据和迁移台账，不以更新快照掩盖失败；
- 合并完成后，各团队再同步集成分支，避免点对点互相合并形成网状依赖。

## 完成定义

团队任务只有同时满足以下条件才可交付集成：

1. 修改路径属于该团队；
2. 输入、输出、失败和能力边界明确；
3. 没有新增跨层导入、厂商对象泄漏或第二套权威类型；
4. 实现和契约假实现通过同一测试，或记录尚未具备替换验证的原因；
5. 精度、回退和性能声明具有对应证据；
6. 相关测试、架构检查和团队范围检查通过；
7. 工作区只包含本任务修改。

## 冲突处理

发现路径归属不清、两个团队需要修改同一实现、或必须改变受保护契约时，停止扩大修改，
由集成负责人先决定权威位置和合并顺序。禁止用复制文件、临时 re-export、自由格式字典或
跨层直调绕过冲突。
