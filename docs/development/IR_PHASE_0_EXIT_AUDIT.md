# FlagQuantum IR Phase 0 退出审计

审计日期：2026-09-01
审计分支：`codex/open-source-api-convergence`
基线提交：`84698f19`
审计结论：**未通过 Phase 0 退出门；Phase 1 实现仍未授权**

关联材料：

- [Phase 0–1 实施计划](../architecture/MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md)
- [Phase 0 事实基线](IR_PHASE_0_BASELINE.md)
- [Phase 0 corpus 设计](IR_PHASE_0_CORPUS_DESIGN.md)
- [Metadata inventory](IR_METADATA_INVENTORY.md)
- [性能基线与预算](IR_PHASE_0_PERFORMANCE_BASELINE.md)
- [Phase 1 实现授权包](IR_PHASE_1_IMPLEMENTATION_APPROVAL_PACKET.md)

## 1. 执行摘要

Phase 0 已经形成有价值的基础资产：公共合同清单、消费矩阵、六份获批 ADR、35-opcode
manifest、metadata inventory、CPU 性能基线和获批的 Phase 1 内部性能预算。专项测试、公共
API 合同测试和默认测试层均可在目标 Docker 环境复现。

本轮技术整改已经关闭原 `IR0-EXIT-001/002`：manifest 中每个语义 oracle 现在都由机器
逐 fixture 分发执行，corpus 已扩展为 12 个正向 fixture、14 个负向 fixture，并覆盖参数
表达式、complex64/complex128、expectation、两参数纠缠梯度、measurement ordering、
routing、QASM/QCIS golden、backend selection、custom 2×2/4×4 matrix 和动态线路拒绝。

扩展 oracle 还发现并修复了一个真实 legacy 缺陷：`CircuitIR(dtype="complex128")` 进入
`fq.run` 时曾被框架默认精度覆盖为 complex64。修复后 CircuitIR dtype 作为 program
constraint 参与 options resolution，并有单元与端到端 expectation oracle 保护。

当前剩余阻塞仅为治理退出门：compiler/runtime/training owner 的最终复核尚未记录，API
owner 也尚未明确授权进入 Phase 1。因此技术门已闭合，但正式 Phase 0 仍保持未通过。

## 2. 本次复现环境

```text
container image: flagquantum-dev:pr-check
platform: Linux 6.12.76 linuxkit aarch64
Python: 3.12.13
Torch: 2.13.0+cpu
dtype: complex64
Torch threads: 1
```

本次没有使用 CPU 结果提出 GPU、分布式或 QPU 能力声明。

## 3. 机器验证结果

### 3.1 Phase 0、文档与公共合同

执行：

```bash
python -m pytest -q \
  tests/internal_ir/test_phase0_corpus.py \
  tests/unit/test_markdown_code_alignment.py \
  tests/unit/test_public_api_snapshot.py \
  tests/unit/test_issue040_ir_public_api.py
```

原始审计结果：`40 passed in 2.88s`。完成技术 blocker 后，Phase 0 corpus 单文件增长为
`52 passed`；加入 ExecutionOptions 精度合同、文档、公共 API snapshot 和 CircuitIR 合同
后的最终组合复核为 `88 passed in 2.49s`。

其中：

- Phase 0 corpus：52 passed；
- 文档、公共 API snapshot 与 CircuitIR 合同：20 passed；
- `CircuitIR` 1.0、稳定根导出和受保护合同未发现漂移。

### 3.2 默认验证层

执行：

```bash
python tools/ci_tier.py pr-default
```

技术整改后的结果：`1019 passed, 10 skipped, 1268 deselected`，耗时 29.45 秒；1 条 PyTorch complex
module 已知 warning，不属于 IR Phase 0 失败。

Runtime integration tier：`168 passed, 31 skipped, 2098 deselected`，耗时 95.88 秒。

### 3.3 性能基线复现

使用原始参数重新执行：

```bash
python benchmarks/internal/ir_phase0_baseline.py \
  --gate-counts 10 100 1000 10000 \
  --iterations 15 \
  --warmup 3 \
  --run-max-gates 1000
```

关键结果：

| Gates | 原 plan p95 | 本次 plan p95 | 原 plan peak | 本次 plan peak |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.190 ms | 0.316 ms | 11,511 B | 11,511 B |
| 100 | 0.850 ms | 0.812 ms | 70,456 B | 70,456 B |
| 1,000 | 6.876 ms | 6.777 ms | 628,493 B | 628,493 B |
| 10,000 | 67.788 ms | 67.534 ms | 6,288,342 B | 6,288,342 B |

10-gate 绝对时间很小，固定开销波动不用于重算预算；100～10K gate 的 plan 结果和所有
峰值内存均复现。1K legacy run p95 本次为 94.36 ms，高于原 64.47 ms，但 legacy run
吞吐不属于 Phase 1 importer+verifier 预算，且本次不覆盖已批准的原始基线。

## 4. P0-001～P0-006 审计矩阵

| 工作包 | 状态 | 已有证据 | 未闭合项 |
| --- | --- | --- | --- |
| P0-001 公共合同保护 | 技术通过、待签审 | 签名、schema、hash、禁改面和 snapshot 测试 | API owner 对事实基线的最终复核未记录 |
| P0-002 消费关系矩阵 | 技术通过、待签审 | 组件矩阵、55-key inventory、typed destination、drift test | compiler/runtime owner 最终复核未记录 |
| P0-003 Characterization corpus | 技术通过、待签审 | 35 canonical opcode、12 正向/14 负向 fixture、serialization、registry drift、参数/dtype/custom matrix | compiler/runtime/training owner 最终复核未记录 |
| P0-004 差分 oracle | 技术通过、待签审 | manifest dispatcher 执行 12 个语义 oracle case，覆盖 state、expectation、gradient、measurement、routing、emitter、backend 和 dynamic rejection | owner 对 oracle 范围与容差最终签审未记录 |
| P0-005 性能基线 | 通过 | 10/100/1K/10K gate、15 iterations、3 warmup、时间和内存，本次已复现 | 无 Phase 0 阻塞；环境变化时不得覆盖原结果 |
| P0-006 Phase 1 性能预算 | 通过 | API owner 已批准内部 importer+verifier 预算，推导关系有测试 | Phase 1 实现后的 budget gate 属于 Phase 1 退出项 |

## 5. ADR 审计

| ADR | 状态 | Phase 1 约束 |
| --- | --- | --- |
| IR-001 | Approved | program semantics 与 execution request 分离 |
| IR-002 | Approved | qubit 使用线性 value 模型 |
| IR-003 | Approved | program、compilation、execution identity 分层 |
| IR-004 | Approved | custom matrix 只支持可验证受限范围 |
| IR-005 | Approved | 参数 identity、tensor dtype 和 late binding 不得漂移 |
| IR-006 | Approved | 只承诺当前静态 CircuitIR 可逆子集 |

六份 ADR 均已批准。批准不自动代表对应实现或退出证据完成。

## 6. Phase 0 正式退出门

| 退出条件 | 结果 | 说明 |
| --- | --- | --- |
| P0-001～P0-006 全部完成 | 技术通过、治理未通过 | P0-001～006 技术证据已闭合，owner review 未闭合 |
| IR-001～IR-006 已批准 | 通过 | 六份 ADR 均标记 Approved |
| corpus 覆盖全部静态 canonical opcode | 通过（仅 opcode 维度） | 35/35；不等于所有声明 oracle 已执行 |
| legacy 语义与性能基线可复现 | 通过 | 语义 dispatcher、runtime/default tier 与原参数性能基线均复现 |
| 公共 API 和序列化合同不变 | 通过 | 专项合同与 pr-default 通过 |
| blocker、owner、目标日期和回滚均记录 | 未通过 | 技术 blocker 已关闭；正式签审 owner 尚未记录结论 |
| API owner 明确批准进入 Phase 1 | 未通过 | evidence 仍为 `phase1_authorized=false` |

## 7. 阻塞项

### IR0-EXIT-001：完成 manifest oracle 执行闭环（已关闭）

已建立按 manifest oracle 分发的参数化测试；未知 oracle 会直接失败。当前 12 个正向
fixture 的所有声明均实际执行，manifest 不再超报 evidence。

### IR0-EXIT-002：补齐最小语义 corpus（已关闭）

已增加独立/重复 Parameter、add/mul/nested ParameterExpression、complex64/complex128
expectation、两参数纠缠解析梯度、measurement ordering、routing legality、QASM/QCIS hash、
custom 2×2/4×4 matrix、backend selection 和 14 个机器可读负向 fixture。Channel 继续按
schema characterization，dynamic 稳定路径以结构化 `CapabilityError` 拒绝。

### IR0-EXIT-003：完成 owner 复核

关闭条件：

- API owner 复核公共事实基线；
- compiler/runtime owner 复核消费矩阵、metadata 分类和 corpus；
- training owner 复核参数 identity 与 gradient oracle；
- 复核结论写入 evidence，不以聊天记录或隐含同意代替。

关闭里程碑：IR0-EXIT-001/002 通过后。

### IR0-EXIT-004：形成明确授权记录

关闭条件：

- 上述 blocker 全部关闭；
- 重新运行专项测试、`pr-default` 和性能基线复核；
- API owner 使用授权包中的精确审批语义批准 Phase 1；
- 更新机器 evidence，但不修改公共 API snapshot 或原始性能预算来制造通过。

## 8. 回滚

Phase 0 当前只增加文档、fixture、测试和内部 benchmark。若不进入 Phase 1，可以删除这些
Phase 0 专用资产；当前 Runtime、Compiler、Stable Core 和公共序列化数据均不依赖它们。

## 9. 最终结论

当前状态可以概括为：

> Phase 0 的技术退出证据已经闭合并可复现，但 owner 最终签审和 API owner 明确授权尚未
> 完成，因此仍不得开始 `_compiler` Phase 1 实现。

下一次审计只需围绕 `IR0-EXIT-003/004` 做 owner 签审和授权复核，不需要继续扩张 corpus、
推翻现有基线或重新设计 ADR。
