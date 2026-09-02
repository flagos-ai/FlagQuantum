# FlagQuantum IR Phase 2 Batch B 实现复核

状态：**Implementation complete — awaiting performance-budget approval**

日期：2026-09-02

## 1. 实现结论

Batch B 已建立私有、版本化的 `universal_rx_ry_rz_cx` target profile。经过 Batch A identity
清理后，全部 30 个静态 unitary opcode 均可确定性降低为 `rx`、`ry`、`rz`、`cx`；channel、
custom unitary 和不满足前置条件的输入 fail closed，不保留非法 gate，也不 fallback。

## 2. 合同与证据

- target gate set 和 certified source set 均写入 pass descriptor；
- 每条 rewrite 显式重连 linear qubit values；
- 中间 ValueId 使用独立 deterministic scope，最终结果复用原 operation result identity；
- 参数缩放使用 symbolic expression，trainable binding 和 autograd 保持；
- 1/2/3 比特 gate 逐项 state differential，允许且只处理 global phase；
- phase/u1/u2/u3、controlled rotation/phase、RXX/RYY/RZZ 逐类 trainable
  forward/gradient differential；
- target legality、幂等、program identity determinism 与 unsupported fail-closed；
- 仅由私有 PassManager 显式调用，默认公共路径不变。

## 3. 独立性能基线

rewrite-heavy workload 的 emitted/source gate ratio 在 10K 时为 6.0：

| Source gates | Emitted gates | p95 | Peak bytes |
| ---: | ---: | ---: | ---: |
| 10 | 48 | 4.098 ms | 121,676 |
| 100 | 589 | 31.978 ms | 1,206,565 |
| 1,000 | 5,984 | 179.677 ms | 12,091,980 |
| 10,000 | 60,000 | 2,059.455 ms | 123,473,111 |

候选预算按至少 `1.25 × measured` 并向上取审查边界生成。该候选尚未获批，因此 Batch B 尚不
具备退出条件。

## 4. 明确未授权

- Batch B exit 与 Batch C routing；
- QASM/QCIS emitter 或 provider codegen；
- public API、默认 compiler/runtime 或 deployment 路径变更；
- legacy compiler retirement；
- 使用此 CPU 基线宣称公开性能或硬件吞吐。

## 5. 下一道门

下一步只能批准 Batch B 内部性能预算并启用机器 gate。该批准不等价于 Batch B exit，也不自动
授权 routing。
