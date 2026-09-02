# FlagQuantum IR Phase 1 实现授权包

状态：**Approved — 仅授权 Batch A 实施**
目标：授权内部 QuantumIR Phase 1，不授权公共 API、默认 Runtime 或 Phase 2 迁移
前置审计：[IR Phase 0 退出审计](IR_PHASE_0_EXIT_AUDIT.md)

实现 owner 技术复核已完成并建议批准，见
[IR Phase 0 实现 Owner 技术复核](IR_PHASE_0_IMPLEMENTATION_OWNER_REVIEW.md)。机器审阅对象、
哈希与未授权字段见 `contracts/ir-phase0-exit-phase1-review-candidate.json`。

## 1. 当前决策

API owner 已于 2026-09-01 使用精确授权口令批准 Phase 0 退出及 Phase 1 Batch A。
不可变的授权记录见 `contracts/ir-phase1-batch-a-authorization.json`。原机器 candidate
继续保留为授权前快照，不回写其 authorization 字段。

```json
{
  "status": "approved",
  "approval": {
    "phase1_authorized": true,
    "batch_a_authorized": true,
    "later_batches_authorized_to_start": false
  }
}
```

本次授权只允许启动 Batch A；完成证据评审前不得启动 Batch B 或修改公共路径。

## 2. 授权对象

获批后只允许新增内部、不可发现的实现：

```text
flagquantum/_compiler/
  diagnostics.py
  identity.py
  ir/
    types.py
    values.py
    operations.py
    modules.py
    schemas.py
    verifier.py
    printer.py
  importers/circuit_ir.py
  exporters/circuit_ir.py
  analyses/
  passes/
```

以及对应资产：

```text
tests/internal_ir/
tests/fixtures/internal_ir/
benchmarks/internal/
docs/architecture/decisions/
docs/development/
```

目录名可以在实现 review 中小幅调整，但依赖方向和不可发现性不得改变。

## 3. 获批后允许完成的工作

### P1-001：类型、Value 与 Module

- 不可变 typed node；
- 确定性 `ValueId/ValueRef`；
- 线性 qubit value；
- `Operation/Block/Region/QuantumModule`；
- source location 与 semantic identity 分离。

### P1-002：Operation schema registry

- operand/result、attribute、region 和 effect 合同；
- 未知 operation、未知核心 attribute 和版本不兼容 fail closed；
- schema 明确声明是否可逆回 `CircuitIR` 1.0。

### P1-003：Verifier 与诊断

- 拒绝 use-before-definition、线性 value 重复消费、旧 value 使用和 release 后使用；
- 拒绝 wire、arity、参数、measurement/result 和 terminator 错误；
- 输出结构化 code、message、location 和 notes，不使用 `print`。

### P1-004：CircuitIR importer

- 接受现有 `CircuitIR`，不要求用户构建内部 IR；
- 保持 opcode、wire order、参数 identity、dtype 和 batch shape；
- 按 IR-001 拆分 program 与 execution request；
- 对 custom matrix、metadata、channel 和 dynamic 标记显式支持或拒绝；
- 输入相同则 internal program identity 相同。

### P1-005：受限 round-trip

- 支持范围内 canonical payload 精确相等；
- 范围外返回结构化诊断，不做 lossy export；
- exporter 仅供测试，不成为 Provider codegen。

### P1-006：Analysis 与 Pass 最小合同

- `DefUseAnalysis`、`QubitLifetimeAnalysis`；
- revision-aware cache 与 preserved-analysis 失效规则；
- 一个无语义变化的示例 Pass；
- 确定性 pipeline digest。

### P1-007～P1-009：差分、正确性和性能门

- 仅测试/开发可启用的新旧路径差分桥；
- state、expectation、measurement、gradient、ordering 和 dtype parity；
- property/fuzz negative tests；
- importer+verifier 通过已批准性能预算；
- 默认 legacy `fq.run/fq.plan` 不加载或执行新路径工作。

## 4. 明确禁止

本授权即使获批，也不允许：

- 修改 `flagquantum/__init__.py` 或 22 个稳定根导出；
- 新增 `fq.compile`、`fq.submit` 或公开 IR/PassManager；
- 修改 `CircuitIR` 1.0 字段、JSON、hash、校验或异常语义；
- 修改 `fq.plan`、`fq.run`、`ExecutionOptions`、`ExecutionPlan` 或
  `ExecutionResult` 的稳定合同；
- 让默认 Runtime、Compiler、Provider 或 DeploymentPackage 依赖 `_compiler`；
- 删除或迁移 legacy compiler/runtime；
- 把 Provider、credential、queue、backend ID 或云任务字段写入程序 IR；
- 以环境变量或 import side effect 静默切换公共执行路径；
- 把 Phase 1 证据描述为 Phase 2、多云、QPU、分布式或性能认证；
- 为使测试通过而更新公共快照、原始基线或已批准预算。

## 5. 实施批次与暂停点

Phase 1 获批后仍分批评审，不能一次性改完再审：

| 批次 | 内容 | 必须暂停评审的证据 |
| --- | --- | --- |
| A | types、values、module、schema | 不可变性、线性模型、deterministic identity |
| B | verifier、diagnostics | 全量负向 fixture 和结构化诊断 |
| C | CircuitIR importer、execution request split | Phase 0 corpus、metadata、参数 identity |
| D | restricted exporter、round-trip | canonical payload 精确相等、范围外 fail closed |
| E | analyses、PassManager、示例 Pass | cache invalidation、pipeline digest、无语义变化 |
| F | differential bridge、correctness、performance | 全量差分、预算、默认快路径零影响 |

每批只在前一批证据通过后进入下一批。批次 A–F 全部通过也只满足 Phase 1 review，不能
自动进入 Phase 2。

## 6. 性能合同

绑定预算：`tests/fixtures/internal_ir/phase1_performance_budget_candidate.json`。

| Gates | Import+verify p95 上限 | Peak host memory 上限 |
| ---: | ---: | ---: |
| 10 | 1.0 ms | 1 MiB |
| 100 | 1.5 ms | 1 MiB |
| 1,000 | 12.5 ms | 1.5 MiB |
| 10,000 | 120 ms | 12 MiB |

预算只适用于指定 CPU 环境的内部 importer+verifier，不是公共 SLA。超预算只能优化、缩小
范围或记录 blocker；不得自动提高预算。

## 7. 必须通过的验证

每个批次先运行 focused tests，再按影响范围运行：

```bash
python -m pytest -q tests/internal_ir
python tools/ci_tier.py pr-default
python tools/ci_tier.py pr-runtime
```

Phase 1 最终还必须证明：

- 35/35 canonical opcode 按声明支持或结构化拒绝；
- manifest 每个 oracle 均有实际机器执行；
- complex64/complex128 state、expectation 和 gradient parity；
- measurement、wire 和 result ordering 精确一致；
- identity、printer 和 pipeline digest 确定；
- 10K gate 时间与内存不超过预算；
- `flagquantum._compiler` 未进入根 namespace、稳定 namespace或用户自动补全；
- 默认 `fq.run/fq.plan` 路径没有新 import 和可测新工作。

## 8. 回滚条件

出现以下任一情况立即停止对应批次并保留 legacy 默认路径：

- 无法保持受支持子集的科学语义或参数梯度；
- 需要修改 `CircuitIR` 1.0 才能继续；
- 新内部路径进入稳定 namespace；
- importer/verifier 无法满足批准预算；
- metadata 无法分类却被静默丢弃；
- 默认 Runtime 出现不可接受回归；
- 新实现需要提前删除 legacy compiler/runtime。

回滚方式：删除 `_compiler`、相关内部测试桥和独立 cache namespace；保留 Phase 0 基线、
ADR、fixture 和测试。公共 API 与现有执行路径不需要迁移。

## 9. 授权前必须关闭

- [x] `IR0-EXIT-001`：manifest oracle 执行闭环；
- [x] `IR0-EXIT-002`：最小语义 corpus 补齐；
- [x] `IR0-EXIT-003`：owner 已通过精确口令接受技术复核；
- [x] `IR0-EXIT-004`：机器授权记录已固化；
- [x] Phase 0 退出已批准；
- [x] 重新运行专项、default 和 runtime 复核且通过。

## 10. 审批语义

技术复核已经完成。具备相应权限的 owner 可以使用以下精确命令，一次性接受
compiler/runtime/training 技术复核、批准 Phase 0 退出并授权 Phase 1 批次 A：

```text
approve IR-PHASE0-EXIT-PHASE1-BATCH-A
```

该命令绑定机器 candidate 中记录的审阅资产和以下完整语义：

> 我接受 compiler、runtime 和 training 技术复核；批准 FlagQuantum IR Phase 0 退出；
> 授权本包定义的 P1-001～P1-009 内部实现。该批准不修改 Stable Core、CircuitIR 1.0、
> 默认 fq.run/fq.plan 或部署合同，不授权 Phase 2、公共 IR API、Provider API 或生产
> 切换，并且只允许先启动实现批次 A。

仅回复普通的“继续”“do”或批准某个性能预算，不自动等价于上述授权。

## 11. Phase 1 完成后的独立审批

Phase 1 实现完成后还需要新的 evidence review。只有 API owner 与 compiler owner 对
P1-001～P1-009、性能预算、默认路径零影响和删除/回滚方案全部签审，才能把 Phase 1 标记
完成。进入 IR Phase 2 必须另行授权。
