# FlagQuantum IR Phase 2 Batch A 完成复核

状态：**Technically complete — awaiting explicit Batch A exit approval**

日期：2026-09-02

## 1. 技术结论

Phase 2 Batch A 的私有静态 canonicalization 已完成实现、语义差分、梯度差分、确定性验证、
性能修复与批准后预算验证。当前技术证据支持进入 Batch A exit 评审，但本文件本身不批准退出，
也不批准 Batch B。

## 2. 已完成范围

- identity 与数值零旋转清理；
- 相邻 self-inverse gate cancellation；
- 相邻 rotation merge，包括 trainable tensor 梯度保持；
- linear qubit value 重连与 revision/identity contract；
- fused private production pipeline 与独立 pass 精确等价；
- entry/exit fail-closed verification；
- immutable identity、pipeline digest 和 schema registry 缓存；
- CircuitIR import → private passes → restricted lowering 全链路差分；
- 10/100/1K/10K successor performance budget gate。

## 3. 批准后的性能门

| Gates | p95 | 上限 | Peak bytes | 上限 | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 0.629 ms | 1.25 ms | 44,480 | 1,310,720 | 通过 |
| 100 | 4.470 ms | 7 ms | 257,326 | 1,310,720 | 通过 |
| 1,000 | 38.603 ms | 65 ms | 2,505,077 | 3,145,728 | 通过 |
| 10,000 | 431.662 ms | 650 ms | 25,075,043 | 31,457,280 | 通过 |

四档 identity determinism 与增长率检查全部通过。这是内部回归预算，不是公开 SLA。

## 4. 保持不变的边界

- Stable Core、CircuitIR 1.0 和 public exports 未改变；
- `fq.run`、`fq.plan`、默认 compiler/runtime 路径未接入新 IR；
- legacy compiler 未删除；
- routing、emitter、provider codegen、TargetIR 和 ProgramIR 未启动；
- 原 Phase 2 budget 快照未修改；
- Batch B 尚未授权。

## 5. 下一道决策门

Batch A 是否退出以及是否只启动 Batch B，必须通过下一份绑定全部证据的 review candidate 单独
批准。未经该批准，不得把“技术完成”表述为“阶段已正式退出”。
