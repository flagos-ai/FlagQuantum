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

### 唯一权威版本

`codex/flagquantum-vnext-architecture` 是唯一集成分支。团队分支是临时开发线，不是发布、
验收或性能声明所引用的产品版本。任何统一版本必须同时记录：

- 集成分支提交 SHA；
- 核心契约和序列化版本；
- 测试与能力证据；
- 对应内部里程碑或标签。

linked worktree 共享同一 Git 对象库，因此团队提交后，集成工作区可以立即按分支名合并，
不需要复制文件，也不要求先推送远程仓库。

### 团队交付前同步

团队必须先提交自己的修改，并确保工作区干净。若集成分支在本轮期间已有新提交，团队在
自己的工作区将集成分支合入当前团队分支：

```bash
git merge codex/flagquantum-vnext-architecture
```

团队只在自己的工作区解决该同步产生的冲突，然后重新运行范围检查、架构检查和模块测试。
团队不得通过 rebase 改写已经交付或被其他会话引用的提交历史。

交付前运行：

```bash
python tools/check_team_scope.py \
  --team TEAM \
  --base codex/flagquantum-vnext-architecture

python tools/check_architecture.py
```

交付记录使用 `TEAM_HANDOFF_TEMPLATE.md`，至少包含共同基线、团队分支、最终提交 SHA、修改
文件、测试证据、能力限制和需要集成团队处理的事项。未提交文件不得作为交付内容。

### 私有仓库下的本地直接合并

集成负责人只在 `FlagQuantum-vNext` 工作区执行合并。开始前确认：

```bash
git branch --show-current
git status --short
```

当前分支必须是 `codex/flagquantum-vnext-architecture`，工作区必须干净。先在对应团队
worktree 复核团队范围检查，再一次只合并一个团队：

```bash
git merge --no-ff codex/vnext-team-compiler
```

合并顺序遵循依赖方向：

```text
Core契约
 -> Compiler与Runtime
 -> Simulation与Platform
 -> Execution Provider
 -> Ecosystem
 -> Agent Services
```

独立或仅含盘点/测试的提交可以调整顺序，但仍须逐个合并、逐个验证。一次合并通过前，
不得叠加下一个团队。

### 每次合并后的门禁

每个团队合入后立即运行：

1. `python tools/check_team_scope.py --validate`；
2. `python tools/check_architecture.py`；
3. 公共API和序列化契约检查；
4. 被合并团队的模块测试；
5. 受影响消费者的测试；
6. 已建立的替换、一致性、精度和回退证据测试。

集成负责人必须检查能力声明、实际执行路径和迁移台账，不能通过更新快照、放宽阈值或新增
静默回退使合并通过。

### 冲突与失败处理

发现路径归属不清、两个团队需要修改同一实现、或必须改变受保护契约时，停止扩大修改，
由集成负责人先决定权威位置和合并顺序。禁止用复制文件、临时 re-export、自由格式字典或
跨层直调绕过冲突。

发生合并冲突时，在集成工作区执行：

```bash
git merge --abort
```

随后由冲突文件的所有者团队同步最新集成分支、解决冲突、测试并提交。集成负责人不得在
不了解算法和证据的情况下替团队重写实现。

合并已经完成但验证失败时，优先由原团队提供修复提交；若必须立即恢复集成分支，使用可
审计的反向提交：

```bash
git revert -m 1 MERGE_COMMIT_SHA
```

已经共享的集成历史不得通过强制推送、hard reset 或重写提交来隐藏失败。

### 团队间同步规则

- 团队分支只能从集成分支获得其他团队的变化；
- 禁止 Compiler 直接合并 Runtime、Runtime 直接合并 Simulation 等点对点同步；
- 一个提交尽量只包含一个团队范围；跨团队功能拆成契约提交和实现提交；
- 一轮集成结束后先产生新共同基线，再通知所有团队同步；
- 团队同步完成前，不开始依赖新契约的下一轮实现。

### 形成新共同基线

整轮测试通过后，记录集成提交 SHA，并创建具有说明的内部标签，例如：

```bash
git tag -a vnext-phase0-integrated -m "FlagQuantum vNext phase 0 integrated baseline"
```

标签只能指向通过本轮规定门禁的集成提交。下一轮任务、性能结果和团队分支同步均引用该
提交或标签，不能使用“最新代码”等不可复现表述。

### 从本地私有开发升级到远程私有协作

早期可以采用：

```text
团队本地提交 -> 集成worktree直接合并 -> 完整验证 -> 推送私有远程仓库
```

真实开发成员和合作单位增加后，升级为：

```text
团队分支推送私有远程
 -> 内部PR
 -> 路径所有者审批
 -> 必需CI
 -> Merge Queue基于最新集成版本复测
 -> 集成分支
```

远程阶段应保护集成分支、禁止直接推送和强制推送，并在获得实际GitHub/GitLab账号或团队
标识后配置 `CODEOWNERS`。私有仓库只限制访问范围，不改变上述版本和合并规则。

## 完成定义

团队任务只有同时满足以下条件才可交付集成：

1. 修改路径属于该团队；
2. 输入、输出、失败和能力边界明确；
3. 没有新增跨层导入、厂商对象泄漏或第二套权威类型；
4. 实现和契约假实现通过同一测试，或记录尚未具备替换验证的原因；
5. 精度、回退和性能声明具有对应证据；
6. 相关测试、架构检查和团队范围检查通过；
7. 交付说明指定一个主领域；普通实现变更若修改多个领域内部，必须说明
   无法拆分的原因和已批准的契约边界；反复涉及四个及以上领域时视为
   架构缺陷，不得直接交付；
8. 新增契约类型已证明现有权威类型无法表达已验证需求；
9. 契约和安全测试之外包含至少一条可读的用户或 Provider 场景测试，或说明
   本变更为何不改变可见场景；
10. 目标领域完成迁移时，其 README 已说明职责、禁止事项、允许依赖、公开入口和
    可在约十分钟内理解、运行、修改并测试的黄金路径；
11. 公共 API 未暴露内部指纹、来源记录、合法性证明、能力快照标识、调度对象或
    证据内部结构；
12. 工作区只包含本任务修改。
