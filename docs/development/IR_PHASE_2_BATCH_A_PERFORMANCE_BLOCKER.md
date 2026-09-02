# FlagQuantum IR Phase 2 Batch A 性能阻塞记录

状态：**Blocked — 不允许 Batch A 退出或进入 Batch B**
日期：2026-09-02
Blocker：`IR2A-PERF-001`

## 1. 结论

Batch A 的授权、四类 canonicalization、线性 value 重连、结构差分、state/expectation
差分和 trainable tensor 双参数梯度差分已经通过，但批准的 Phase 2 总管线预算不可达。

预算没有修改，Batch A 不得宣布完成。

## 2. 机器事实

Docker CPU 环境中的 1K gate 最小有效采样：

| 指标 | 实测 | 已批准上限 | 结果 |
| --- | ---: | ---: | --- |
| 完整管线 p95 | 67.629 ms | 15.0 ms | 失败 |
| Peak host memory | 2,726,417 bytes | 2,097,152 bytes | 失败 |
| identity determinism | true | true | 通过 |

分段冷启动诊断：

| 阶段 | 1K gate 单次耗时 |
| --- | ---: |
| seal/import/verify | 17.971 ms |
| Batch A pass pipeline | 52.180 ms |
| optimized lowering/verify | 16.778 ms |

完整 gate 连续两次没有产生可接受的完整 10K evidence，因此未继续重复同一命令。1K gate
已经同时超过时间和内存上限，足以触发 fail-closed blocker。

## 3. 根因

预算推导使用约 `2 × legacy_plan p95`，但比较范围定义为：

```text
CircuitIR import + verify
  + Phase 2 passes
  + restricted lowering
  + lowered-result verification
```

1K gate 的 Phase 1 import/verify 批准上限本身就是 12.5ms，因此 Phase 2 的 15ms 总预算只给
新增 pass、lowering 和再次验证留下 2.5ms。这不是通过微调实现可以可靠满足的边界。

首轮实现还发现并修复了 cancellation/rotation 对输出列表的二次扫描，将 last-touch 查询改为
按 wire 维护历史栈；优化后 1K gate 仍为 67.629ms，证明剩余问题主要是预算口径和多次
identity/verification/serialization 成本，而不是继续删除一个局部循环即可闭合。

## 4. 已通过的非性能证据

- 精确授权记录绑定原 review candidate；
- identity、零旋转、自逆门、相邻旋转与 disjoint-wire 行为匹配 legacy；
- 每个独立 pass 幂等；
- trainable rotation merge 保持 forward 和两个原参数梯度；
- complex64/complex128 state 与 expectation parity；
- malformed multi-block 和缺失 source provenance fail closed；
- 默认路径没有接入 `_compiler`；
- 原预算文件保持不可变。

这些证据不覆盖或替代性能退出门。

## 5. 建议 remediation

下一步需要独立授权完成：

1. 分别测量 import、identity、各 pass、verification、lowering 和 serialization；
2. 缓存 immutable module identity，避免同一 revision 重复 JSON/hash；
3. 将完整验证从“每个 by-construction pass 重复执行”收敛为入口和出口各一次，同时保留每个
   pass 的独立 verifier 测试；
4. 评估 fused production pipeline 与独立 pass descriptor 的一致性；
5. 采集 10/100/1K/10K 稳定结果；
6. 基于组成成本提出 successor budget，并由 owner 单独批准；
7. 不修改 Phase 0、Phase 1 或原 Phase 2 candidate 预算快照。

## 6. 暂停边界

在 remediation 和 successor budget 获批并通过前：

- 不进入 Batch B；
- 不接入默认 compiler/runtime/deployment；
- 不更新原预算制造绿灯；
- 不宣称 Phase 2 性能或 Batch A 完成；
- 不删除 legacy compiler。

建议精确授权口令：

```text
approve IR-PHASE2-BATCH-A-PERFORMANCE-REMEDIATION
```

该口令只授权性能诊断、内部优化和 successor budget 候选，不自动批准新预算，也不授权
Batch A 退出或 Batch B。
