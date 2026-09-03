# ARCH-008：迁移决策账本与快速/标准/认证路径

状态：Proposed

日期：2026-09-03

依据：Phase 0 重复类型与兼容债务盘点；不授权迁移、退役或公共 API 变化

## 上下文

并非每个重复类型都应立即迁移，也并非每个局部优化都需要穿过 Core、Compiler、Runtime、
Simulation、Provider 五层。缺少成本与期限的“兼容”会变成永久双轨；反之，强制所有改动走
完整认证路径会拖慢本地 CPU、单卡和局部 kernel 工作。

## 决策候选

每个跨层或 legacy 项在行动前建立迁移决策记录，至少包含：当前/目标 owner、调用者与事实
权威、用户/架构价值、持续维护成本、一次性迁移成本、回归风险、继续存续的影响、兼容承诺、
复核时间、目标版本、测试证据和退出条件。允许的 disposition 只有：

- `migrate`：迁到新权威；要求兼容计划、替换测试和旧调用者归零；
- `adapt`：保留实现但通过窄 adapter 接入权威合同；要求损失性说明、owner 和退出/复核条件；
- `freeze_legacy`：当前保留且禁止新增调用者、功能和 schema 扩张；必须有 owner、characterization/
  no-new-caller 测试、明确复核日期及届时重新选择 disposition；
- `retire`：只有替代路径、兼容窗口、调用者归零和批准的移除测试齐备后执行。

`freeze_legacy` 不是无限兼容层；逾期未复核即阻塞依赖它的新功能和发布晋级，不能自动续期。

### 三条验证路径

1. **快速路径**：模块内部或已有语义下的局部 kernel/性能优化；不改变公共语义、跨层合同、
   capability 上限或 fallback。由 owner 做范围/架构检查和最小数值、梯度、性能回归测试，
   不强制跨五层。
2. **标准路径**：新增/改变跨层信息、内部版本化合同、adapter 或 provider 行为；走 ADR/contract
   fake、消费者替换、兼容和相关集成测试。只有公共语义变化才进入 Core/API change 流程。
3. **认证路径**：硬件、分布式扩展性、生产、性能或 QPU 声明；在标准路径之上运行真实目标、
   对应 GPU/多节点/Provider 测试、evidence audit 和 release gate。接口或 Mock 不得替代。

## 禁止事项

- 不用“临时兼容”隐藏无 owner、无成本评估、无复核时间的永久双轨。
- 不以搬文件、改 import 或更新 snapshot 代替替换/兼容证据。
- 不因局部实现使用新 kernel 就提升公共 capability，或强制无语义变化的实现修改 Core。
- 不让快速路径绕过已存在的 public API、fail-closed、fallback 和数值正确性门禁。

## 兼容性

本 ADR 只定义决策资料和验证强度，不改变已有 deprecation、Stable Core 或 release policy。
涉及稳定导出、签名、行为、异常或序列化的 `migrate`/`retire` 仍需 API Change Proposal；仅有
ADR 批准不足以授权。现有 legacy 默认保持原状，直至逐项记录和批准。

## 迁移顺序

1. 为 Phase 0 D1–D12 和 Runtime→Compiler 清单补齐上述字段及 disposition。
2. 先处理高价值、低回归风险的 adapter/消歧，再处理身份和跨层权威迁移。
3. 每次只迁移一个纵向切片，验证后更新调用者计数、维护成本和复核时间。
4. freeze 项到期复审；retire 项按兼容窗口和批准顺序删除。

## 验收测试

- 决策记录 schema 必填六类评估、owner、review date、tests 和 exit condition；
- freeze 项禁止新增 importer/caller/export/feature，且 characterization 保持；
- migrate/adapt 的 fake 与真实实现通过相同 conformance，消费者替换不改代码；
- 快速路径证明 public contract/capability 不变；认证路径缺真实 evidence 时 fail closed；
- CI 能识别逾期 freeze、无 owner adapter 和未满足退出条件的 retire。

## 未决问题

- 迁移台账最终采用 TOML/JSON 的位置、schema owner 和逾期阻断级别；
- 价值/成本/风险使用枚举、评分还是带依据的文本；
- 快速路径性能回归阈值和何时因可观察行为变化自动升级为标准路径。
