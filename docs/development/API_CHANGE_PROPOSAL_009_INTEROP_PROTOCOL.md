# API Change Proposal 009：框架无关互操作协议

## 状态

**Implemented, pending API-owner review — 已实现，尚未冻结。**

- 候选稳定命名空间：`flagquantum.interop`；
- 根级名称变化：无；
- 机器可读契约：`contracts/interop-protocol-v1-candidate.json`；
- 实施授权：API owner 于 2026-09-01 要求继续执行稳定化漏斗；
- Qiskit、PennyLane 具体适配器仍为 experimental；
- 本提案不代表整个首次公开 Alpha API 已冻结。

## 决策

稳定“如何接入外部量子框架”的协议，不稳定“某个第三方版本的具体实现”。候选协议包含：

- `InteropAdapter` 与版本协商；
- 不可变、懒加载的 adapter registry；
- 统一 import/export result 与机器可读 conversion report；
- 严格转换默认 fail closed，显式 `allow_lossy=True` 才允许有损转换；
- 框架无关 round-trip、rejection 和 semantic fingerprint conformance；
- 与 `flagquantum.errors.FlagQuantumError` 对齐的异常边界。

以下不进入候选稳定清单：`DEFAULT_INTEROP_REGISTRY`、`qiskit`、`pennylane` 以及所有
adapter-specific 类型和函数。新增或稳定某个 adapter 必须单独提案并声明版本窗口。

## 稳定边界

```text
external framework object
        │
        ▼
experimental adapter implementation
        │  InteropImportResult / InteropExportResult
        ▼
candidate-stable flagquantum.interop protocol
        │
        ▼
versioned CircuitIR
```

外部对象不能进入 compiler、runtime、kernel、distributed 或 accelerator 层。导入
`flagquantum.interop`、查看 registry 或加载 adapter 描述符不能隐式导入 Qiskit/PennyLane。

## 本轮修正

1. `InteropError` 进入稳定 `FlagQuantumError` 体系，同时保留 ImportError、ValueError、
   RuntimeError 的 Python 兼容分类；
2. `run_adapter_conformance` 的 callable 默认值改为 `None`，消除签名中进程地址导致的
   不可复现契约；
3. `flagquantum.interop.__all__` 排除具体 adapter 和默认 registry 实例；
4. `fq.experimental.interop` 只路由 `qiskit`、`pennylane` 两个实验实现命名空间。

## 验收标准

- [x] 候选稳定导出与机器契约完全一致；
- [x] 稳定根 `fq.__all__` 不变；
- [x] 错误类型统一且保留内建异常兼容；
- [x] 无第三方依赖环境可以导入、发现并审查 adapter；
- [x] Qiskit 2.0/2.5 与 PennyLane 0.44/0.45 的适配器证据继续独立运行；
- [x] 具体适配器未被误标为稳定；
- [ ] API owner 批准冻结框架无关 interop protocol。
