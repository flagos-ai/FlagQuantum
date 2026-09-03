# ARCH-001：Provider 分层与 Core 契约所有权

状态：Approved
日期：2026-09-03
适用范围：FlagQuantum vNext 目标架构与分阶段迁移

批准记录：项目负责人通过明确指令批准落实本决策。该批准确立目标边界和防止新增架构
债务的检查，不授权立即移动稳定 API、删除现有实现或宣称目标能力已经完成。

## 背景

原候选架构把 Accelerator、Simulation、QPU 和 Remote Service 并列为 Provider，但它们
不处于同一抽象层级。Accelerator 是执行引擎使用的计算平台，Simulation、QPU 和 Remote
Service 则接受完整执行请求。同时，原机器契约允许 Runtime 依赖 Compiler Contracts，
可能使编译器内部类型成为运行时的长期依赖。

当前代码仍存在 `_compiler`/`compilation`、`simulation`/`runtime/backends`、
`runtime/platforms`/`extensions/sdk` 等并行权威入口，迁移若无完成证据和退出条件，可能
形成永久双轨结构。

## 决策

### 1. Provider 分为两个层级

- **Execution Provider** 接受 Core 定义的完整执行请求，包括 Simulation、QPU 和 Remote
  Service Provider；
- **Platform Provider** 提供设备、Kernel、精度和通信能力，包括 CPU、国产 GPU/NPU
  和通信实现；
- Simulation Execution Provider 组合 Simulation Engine 与 Platform Provider；
- QPU 和 Remote Service Provider 不因共享 Provider 名称而依赖模拟算法。

### 2. 跨领域契约由 Core 唯一所有

ProgramArtifact、TargetCapabilities、ExecutionRequest、ExecutionResult、Evidence 以及
Execution/Platform Provider Contract 均由 Core 定义。Compiler、Runtime、Simulation 和
Provider 只能消费或实现这些契约，不得复制定义。

目标态 Runtime 不依赖 Compiler 包。Compiler 输出 Core 定义的产物，Runtime 只消费该
产物。当前 Runtime 到 `_compiler` 或 `compilation` 的导入登记为封闭迁移清单，新导入由
架构检查器拒绝，现有条目只能减少。

### 3. 所有迁移项必须可退出

每个迁移项必须在机器可读台账中登记：

- 责任团队与目标里程碑；
- 当前和目标权威位置；
- 兼容适配器；
- 完成证据；
- 旧实现退出条件；
- 当前状态。

不存在退出条件的目标目录不得作为平行实现长期建设。

## 备选方案与否决原因

### 所有外部实现使用一个通用 Provider 接口

否决。设备平台和完整执行目标的生命周期、输入输出及失败语义不同，统一接口会产生
大量可选方法、类型判断和厂商分支。

### Compiler 自己拥有 Executable 和 Target 契约

否决。Runtime 将不得不导入 Compiler 类型，编译器替换会传导至执行层。

### 先建立全部新目录，再逐步填充

否决。会在没有权威切换条件时形成两套实现，增加多人协作冲突和维护成本。

## 兼容性与迁移

- 本决策当前只改变候选架构契约、文档和架构检查，不改变 Stable Core 公共 API；
- 现有 Runtime→Compiler 导入暂时保留在有界清单中；
- `runtime/platforms` 与 `extensions/sdk` 在新 Platform Provider Contract 获得批准和替换
  证据前仍是当前权威入口；
- 每次迁移只处理一条完整纵向路径，通过后才关闭对应旧入口。

## 验收

- [x] 机器架构契约区分 Execution Provider 与 Platform Provider；
- [x] 机器架构契约指定 Core 为跨领域契约唯一所有者；
- [x] Runtime 目标依赖中不再包含 Compiler Contracts；
- [x] 架构检查器禁止新增 Runtime→Compiler 导入；
- [x] 七条迁移轨道均有责任团队、里程碑、完成证据和退出条件；
- [ ] Simulation Engine 在两个 Platform Provider 间完成替换验证；
- [ ] Runtime 在 Simulation 与 QPU/Remote Service Provider 间完成替换验证。
