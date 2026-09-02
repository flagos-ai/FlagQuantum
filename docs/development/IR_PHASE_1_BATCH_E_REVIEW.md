# FlagQuantum IR Phase 1 Batch E 评审

状态：**Technical complete — 等待 Batch E 退出评审**
日期：2026-09-02
实现范围：AnalysisManager、DefUse/QubitLifetime、PassManager 与示例 Pass

## 1. 结论

Batch E 已建立私有 QuantumIR 的最小分析与 Pass 基础设施。它可以确定性遍历 IR，按
program identity 与 revision 缓存分析结果，并通过显式 preserved-analysis 声明安全复用
跨 revision 的结果。

本批没有迁移现有优化器，也没有让默认 `fq.run/fq.plan` 进入新 IR 路径。

## 2. Analysis 合同

- `DefUseAnalysis` 记录 block argument、operation result 与 operand use；
- `QubitLifetimeAnalysis` 记录线性 qubit value 的定义、消费、对应 successor 与终止状态；
- 多 qubit operation 按 operand/result 位置一一延续 lifetime，不把一个输入错误映射到全部输出；
- cache 归属于 `AnalysisManager` 实例，不使用进程级全局状态；
- cache key 绑定 analysis 名称、版本、program identity 与 module revision；
- 只有 Pass 明确声明 preserved 的分析才会被带入新 revision，其他分析在新 revision 自动 miss。

## 3. Pass 合同

`PassDescriptor` 将名称、版本、options、seed 和语义保持声明纳入确定性编码；流水线 digest
同时绑定 Pass 顺序。`PassResult` 统一返回 module、changed、preserved analyses、diagnostics
与 statistics。

PassManager fail closed：

- Phase 1 Pass 必须声明语义保持；
- program identity 变化直接拒绝；
- 声明 changed 却未推进 revision 直接拒绝；
- 声明 unchanged 却返回不同 module 直接拒绝；
- error diagnostic 停止后续 Pass，失败输出不成为当前 module；
- warning/note 可以保留在结果中，但不被误判为 pipeline failure。

示例 `CanonicalizeAttributesPass` 仅检查已不可变、已 canonical 的 attribute 表示，因此保持
同一 module 对象和 program identity。它用于证明合同，不代表已有实际优化 Pass。

## 4. 能力边界

当前已经具备“安全承载后续内部 Pass”的底座，但尚不具备：

- 语义优化、分解、布局或路由 Pass；
- legacy executor 与 QuantumIR 的差分执行桥；
- state、expectation、measurement 或 gradient parity 认证；
- importer/verifier 性能预算认证；
- 公共 IR、PassManager 或插件 API；
- 默认或生产执行路径切换。

## 5. 验证结果

| 检查 | 结果 |
| --- | --- |
| Internal IR | 193 passed |
| Default CI | 1160 passed, 10 skipped |
| Runtime CI | 168 passed, 31 skipped |
| Ruff | passed |
| Black | passed |

Default CI 保留一条既有 PyTorch complex module warning；Runtime 的 31 个 skip 为既有
环境/硬件条件跳过。机器证据和精确文件 hash 见
`contracts/ir-phase1-batch-e-review-candidate.json`。

## 6. 下一暂停点

如 owner 接受本证据并希望启动 Phase 1 最后一批 Batch F，应使用：

```text
approve IR-PHASE1-BATCH-E-EXIT-BATCH-F
```

该口令只授权测试/开发差分桥、正确性与梯度 parity、property/negative testing、已批准的
importer/verifier 性能预算验证，以及默认快路径零影响证明；不授权公共 API、生产路径切换、
Phase 2、Provider codegen 或现有优化器迁移。
