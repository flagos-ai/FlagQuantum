# ARCH-001：Compute、Remote 与 Core 契约所有权

状态：Approved
日期：2026-09-03
修订：2026-09-08
适用范围：FlagQuantum vNext 目标架构与分阶段迁移

## 背景

以 Provider 为总称会把两类生命周期完全不同的对象混在一起：本进程直接控制的
CPU/GPU/NPU，以及必须通过外部控制面提交任务的 QPU、GPU/HPC 服务和云平台。
按硬件种类或厂商划分无法稳定表达这一区别，也容易产生重复注册、循环依赖和模糊
的公共接口。

## 决策

### 1. 按控制边界划分

- **Compute**：当前进程直接发现、激活和调用的算力，负责设备生命周期、精度、内存、
  Kernel、通信和执行路径事实；
- **Remote**：通过外部任务控制面调用的算力，负责目标发现、凭据、提交、状态、取消和
  结果解码；
- Simulation 实现数值算法且不导入 Compute；Runtime 组合 Simulation 与 Compute，或选择
  Remote，并管理执行生命周期；
- 同一型号 GPU 可以因部署方式不同而属于 Compute 或 Remote；真实 QPU 通常属于
  Remote。

### 2. 跨领域契约由 Core 唯一所有

ProgramArtifact、TargetCapabilities、ExecutionRequest、ExecutionResult 和 Evidence 由
Core 定义。Compiler、Runtime、Simulation、Compute 与 Remote 只能消费或实现这些契约，
不得复制定义。目标态 Runtime 不依赖 Compiler 内部类型。

### 3. 所有迁移项必须可退出

迁移项必须登记责任团队、目标位置、完成证据和旧实现退出条件。不存在退出条件的新
目录不得与旧权威长期并存。

## 备选方案与否决原因

- **统一 Provider 接口**：否决。直接设备生命周期与远程任务生命周期不同，统一接口
  会产生大量可选方法和类型判断。
- **按 CPU/GPU/QPU 分类**：否决。GPU 既可本地直控也可通过远程服务使用，硬件类型不
  能表达控制关系。
- **将远程 GPU 服务命名为 Cloud**：否决。远程不等于公有云，也可能是九鼎算力、内网
  集群或本地机房控制面。

## 兼容性与迁移

- 代码尚未正式发布，旧 `flagquantum.providers` 路径直接删除，不保留转发层；
- 原设备平台实现迁入 `flagquantum.compute`；
- 外部 Braket、Quafu 和通用 HTTP 任务适配迁入 `flagquantum.remote`；
- 每次迁移处理完整调用路径，通过测试后关闭旧入口。

## 验收

- [x] 机器架构契约区分 direct compute 与 remote compute；
- [x] Core 是跨领域契约唯一所有者；
- [x] 代码目录不再保留 `flagquantum.providers`；
- [x] 架构检查器禁止该旧目录返回；
- [ ] Compute 的公共类型去除遗留 Provider 命名；
- [ ] Remote 与 Deployment 的共享任务契约完全收敛到 Core。
