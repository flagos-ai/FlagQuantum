# ARCH-005：Execution Result 与 Evidence 兼容边界

状态：Proposed

日期：2026-09-03
依据：Phase 0 八团队盘点；不新增或替换结果实现

## 上下文

稳定 `ExecutionResult`、Core `ExecutionRecordContract`、Deployment/Target/Dynamic/Compiler ABI
结果及多种 backend result 并存。Runtime evidence、Core provenance、capability evidence、审计
verdict 与自由 metrics 也未统一。数值结果、持久执行记录和发布声明是不同职责，直接合并会
丢失位序、身份、失败、回退或真实执行路径。

## 决策候选

1. 稳定 `ExecutionResult` 继续是用户投影；本 ADR 不移动其实现、不承诺 tensor payload
   序列化，也不改变现有 accessor。
2. Core 另拥有版本化 Execution Record/Evidence envelope，记录 request/plan/artifact/attempt/
   provider identities、实际 target/path/device/precision、ownership、memory、communication、
   fallback/degradation、timestamps、failure 和 namespaced evidence references。
3. 原始测量、签名材料与审计 verdict 分层保存并由 digest 关联；audit 决定 claim eligibility，
   Provider 或接口存在不能自行声明能力。
4. backend-native result/metrics 可保留在域内；跨 Provider 边界必须无损投影核心字段，扩展值
   进入受控 namespace，不能覆盖核心字段。

## 禁止事项

- 不创建第四个公共 result，不把 counts 直接包装为“统一结果”而遗漏 wire/bit order。
- 不把自由 metadata 当作 Evidence，不让 native 字段覆盖 failure、fallback 或 identity。
- 不以 CPU distributed、Mock、replicated execution 或接口测试形成 scalability/QPU 声明。
- 不通过修改快照承诺新的 tensor 序列化或改变稳定异常。

## 兼容性

现有 result 保持字节和行为兼容；adapter 只补充旁路 record/evidence。旧 provenance/metrics
读取规则维持，冲突必须可见。若未来向稳定 result 增字段、改类型或改 summary/diagnostics，
必须使用新 schema/version、迁移 fixture 和 API Change Proposal。

## 迁移顺序

1. 冻结稳定 result、Deployment counts、bit-order 与旧 evidence fixture。
2. 定义 evidence envelope/failure taxonomy 和 contract fake。
3. 先接 Local Simulation adapter，再接 Remote/QPU fake adapter。
4. Runtime attempt coordinator 统一成功与失败 record；Audit 消费同一 envelope。
5. 调用者归零且兼容窗口结束后退出 Deployment/Target 跨层结果。

## 验收测试

- stable result 行为/序列化 fixture 不变；核心字段冲突不可覆盖；
- 非对称 bit order、leading zero、shot accounting、失败与取消终态测试；
- success/failure/fallback 均产生 identity 完整的 record；
- local simulation 与 remote fake 通过同一 result/evidence conformance；
- distributed claim 缺任一规定字段或真实 evidence 时 fail closed。

## 未决问题

- Execution Record 的公开程度、存储期限与隐私删减规则；
- HMAC/signature key 管理和 artifact URI 归属；
- 大 tensor/sample payload 的引用格式、metrics namespace 注册及 failure cause 链上限。
