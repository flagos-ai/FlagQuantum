# FlagQuantum IR Metadata Inventory

状态：Phase 0 静态 inventory 与 importer-relevant typed destination 已建立
机器清单：`tests/fixtures/internal_ir/metadata_inventory.json`

## 1. 方法与边界

静态 AST 扫描发现 55 个被 `metadata`-like mapping 通过字符串字面量读取/写入的 key。
扫描覆盖 `flagquantum/**/*.py` 的 `get/pop/setdefault/subscript`，并由测试检测漂移。

该数字不是“CircuitIR 有 55 个公共字段”：同名 metadata 容器存在于 Instruction、
MeasurementNode、CircuitIR、ExecutionResult、runtime evidence 和 DeploymentPackage。
Inventory 的目的正是防止 importer 把这些不同作用域混为一谈。

静态扫描也不是完整语义分析。由 dict merge、变量 key、外部 payload 或仅生产未读取的
key 需要人工补充；当前已单独记录 `logical_state_shape`。

## 2. Importer 相关分类

- **Instruction semantics**：channel、dynamic/condition、classical bit、global phase、
  diagonal/MPO/Pauli/Schmidt hint；
- **Execution request**：seed、postselection、measurement format/name 和资源上限；
- **Constraints**：batch size、runtime config，以及人工补充的 logical state shape；
- **Compiler evidence**：routing、routing strategy 和 dependency scheduling evidence；
- **Interop 待复核**：circuit name、interop、num_clbits、Qiskit label；
- **不属于源 CircuitIR**：deployment/provider payload、runtime reverse records、claimability
  与 platform evidence。

完整 key 和分类以机器清单为准。

## 3. Phase 1 规则

1. 先确定 metadata 所属对象，再判断语义；
2. 影响执行、合法性、结果、梯度或 identity 的 key 必须进入 typed destination；
3. routing/provider/runtime evidence 不进入 program semantic hash；
4. 未分类 key 不得静默丢弃或自动归为 provenance；
5. dynamic/condition 在 `circuit_ir_v1_static` profile 中结构化拒绝；
6. inventory 漂移测试失败时必须分类新 key，不能直接更新期望集合；
7. 本 inventory 不冻结 public metadata API，也不把内部 key 升级为兼容承诺。

## 4. 未完成事项

- [x] repository-wide 静态 consumed-key inventory；
- [x] importer 相关与外部 metadata 初步分层；
- [x] 自动 drift test 输入格式确定；
- [x] 对变量 key、dict merge 和 provider payload 完成人工二次审计；
- [x] 为每个 importer-relevant key 定义 typed destination；
- [x] interop 四个待复核 key 完成 semantic/provenance 决策；
- [ ] compiler/runtime owner 批准最终分类。
