# Target Capabilities v1 Compiler Adapter Handoff

状态：Compiler 内部最小适配；不改变 Stable API、旧 schema、默认路径或失败阶段

## 边界

本适配器只把私有 `_compiler.TargetCapabilities` 投影为 Core-owned `RequirementSet`。
Compiler 负责从程序和目标合法性产生需求，不负责发现设备、生成平台事实或认证运行证据，
因此产品代码不提供从 Compiler 对象生成 `TargetCapabilitySnapshot` 的入口。

可直接投影的字段包括 target class、逻辑/物理量子比特容量、measurement results、artifact
profiles、shots/operations limits 和最大 Compiler ancilla 数。所有 fallback 授权保持缺省
false；适配器不选择目标、不执行 fallback，也不改变 Runtime 行为。

## 损失核算与旧权威

以下字段尚未进入 Core v1 闭集，始终生成 typed loss record，并继续由现有
`compare_target_capabilities()` 裁决：topology、control flow、mid-circuit measurement、reset、
timing、pulse、noise、parameter binding 和 calibration hash/validity。即使字段处于缺省或
inactive 状态也保留记录，避免迁移期间把“没有提出要求”误写成“字段不存在”。

`gates.native` 的参数域覆盖和 `ancillas.policy` 的等级覆盖比 Core v1 通用 `covers` 更丰富；
二者不进入可独立匹配的 Core `RequirementSet`，而是生成 `legacy_comparator_required` 记录。
适配结果以 `requires_legacy_comparator` 明示 Core 子集不是完整合法性裁决，并通过
`compare_available()` 调用不变的旧 comparator。这样，合法的更宽门参数域或更高 ancilla 等级
不会被 Core 的保守通用比较误拒绝，同时这些要求也不会被静默丢弃。

适配结果保存旧 semantic payload 的 canonical JSON 和原 fingerprint，可恢复旧对象并验证
fingerprint 不变。`display_label` 继续是非语义字段，不进入保存的 semantic payload。

## 生命周期

- owner：Compiler；
- adapter version：1.0；
- supported legacy schema：`target_capabilities_v1`；
- 可观测量：typed loss 数量、active loss 数量、Core/legacy comparison divergence；
- 退出条件：所有 Compiler consumer 均迁移到批准的 Core contract，且旧 importer 使用量为零，
  同时 golden schema、fingerprint 和 compatibility fixtures 全部保持通过；具体删除版本须由
  Integration 另行批准。

本提交不修改旧 `TargetCapabilities`、`semantic_fingerprint`、comparator、默认值、公共导出、
部署/Runtime 调用链或任何历史 contract。
