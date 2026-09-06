# vNext Phase 2 Target Capabilities Integration 裁决

状态：Integration 裁决；授权后续内部最小实现，不是 Stable API 发布

日期：2026-09-03

## 最终结论

三份团队提案合并为一个 Core-owned 谱系：`CapabilityRequirement` 是单谓词，
`RequirementSet` 是谓词与 fallback 授权集合，`TargetCapabilitySnapshot` 是有身份、scope 和
有效期的事实快照。Platform 的 candidate 只是 adapter 输入，不保留第四类型；Evidence 引用和
execution observation 分别属于证据引用与一次执行记录，也不属于 capability 类型。

这项裁决选择“稀疏谓词 + 通用 fact envelope”，而不是把 Core、Platform、Runtime 三份字段
拼成巨型 dataclass。这样保留当前 `_compiler.TargetCapabilities` 已验证的 gates、parameter
domain、result、artifact、capacity、limit 和 ancilla 覆盖语义，同时避免把 dynamic、checkpoint、
realtime、topology、communication 等仍不稳定域提前冻结。

## 精确 v1 形状

`CapabilityRequirement`：`name`、`operator`、`value`、`strength`、`source`、
`minimum_evidence_level`、`accepted_exposures`。

`RequirementSet`：`schema_version`、`requirement_set_id`、`requirements`、
`fallback_authorizations`、`extensions`。

`TargetCapabilitySnapshot`：`schema_version`、`snapshot_id`、`target_identity`、`scope`、
`captured_at`、`valid_until`、`facts`、`evidence_refs`、`blockers`、`extensions`。

每项 fact：`name`、`value`、`support_status`、`fact_exposure`、`source`、`blockers`。每项 source：
`kind`、`ref`。每项 evidence ref：`evidence_id`、`sha256`、`level`、`scope`。

Target identity 固定为 `target_id`、`target_class`、`provider`、`provider_version`、
`target_revision`、`environment_id`。Scope 固定为可空的 `device_ids`、`dtype`、`kernel`、
`workload_id`、`world_size`、`node_count`。所有 Core-owned object 拒绝未知字段/enum/version；唯一
扩展点是 `extensions`。

v1 operator 为 `equals`、`at_least`、`at_most`、`contains_all`、`covers`。Evidence level 为
`basic < observable < certification`。Support 与 exposure 严格正交：
`unknown/unmeasured/unsupported/verified` 回答支持结论，
`observed/declared/not_exposed/unknown/not_applicable` 回答值的来源或缺失原因。

所需 evidence threshold 取所有适用 mandatory requirements 与 claim gate 的最强等级；候选
evidence ceiling 取所有不可缺 evidence references 的最弱等级。只有 ceiling 不低于 threshold，
且 status、accepted exposure、freshness、scope 和值比较同时满足时 mandatory 才通过。
`verified/declared` 仅用于机器授权中 `authoritative_static_declaration_allowed` profile 明确列出的
权威静态规格；裸声明默认仍为 `unmeasured/declared`，容量、precision、延迟、物理 route 和
“无 fallback”必须 observed。Preference 只排序
可执行候选。任何缺失、stale、unknown、unmeasured、unsupported、not_exposed、无依据
not_applicable、未知 extension handler 都 fail closed。

## Precision 与 fallback

Precision 不是一个字符串。v1 分为 native、effective、storage、parameter、accumulator dtype 与
software mechanism 六个谓词。Double-Single 是 software mechanism；它可在窄 scope 证据下支持
effective precision，但永远不能冒充 native FP64/complex128。

Fallback 授权固定为 backend、device、CPU、precision、algorithm、approximation 六轴，缺省
全部禁止。每次 fallback 创建新候选并用其自身 snapshot 全量重匹配。CPU 是正常的独立候选，
不是 resolver 默认值；宽泛 backend fallback、preference 或 policy 不能推导 CPU fallback。

## 所有权裁决

- Compiler：程序/target legality 和 compiler-source requirements；不发现平台，不授权 fallback。
- Platform/Execution Provider：发现平台或 target 事实，给出 source/evidence references；不选择。
- Runtime：合并来源但不改写来源，匹配、排序、租约、placement 和 fallback decision。
- Simulation：算法合法候选、资源/成本/误差提示；不把估算写成平台事实。
- Runtime + executing Provider：一次 attempt 的实际路线、精度、设备、fallback 和分布证据。
- Audit/Release：claim eligibility；接口、Mock 或 provider 声明不能替代认证。

## v1 闭集与延后项

首批闭集仅覆盖 target/device/count/memory、qubit/gate/measurement/artifact/limit/ancilla，以及
precision 六轴。dynamic、checkpoint/restart、realtime/session、完整 topology/placement/link、
P2P/collective/physical route、calibration lifecycle、gradient/optimizer distribution 和完整
workload language 全部延后。

延后域只能先进入登记的 namespaced extension，并具备 owner、版本、matcher、测试和晋级/退出
条件；未知 handler 失败关闭。它们稳定后才通过后续 contract proposal 进入 Core 闭集。这既
保留演进路径，也避免 v1 变成包含所有未来场景的通用规则引擎。

## 授权与禁止

本裁决授权后续提交实现内部 Core 值对象、strict serialization、canonical identity、纯 matcher、
contract fake/conformance，以及 `_compiler.TargetCapabilities`、CPU Platform、Runtime backend、
Deployment/Execution Provider 的窄 adapters。adapter 不可表达的旧字段必须保留 blocker 并继续由
旧权威处理。

本裁决不授权当前提交修改产品实现，不授权 Stable Core 导出、旧 public/schema/fingerprint、
plan/result、默认 backend/fallback、failure stage 或 capability maturity 的变化；不授权真实网络、
provider/hardware 接入，也不构成国产硬件、FlagCX、多节点、QPU、生产或 scalability 验证。

## 后续实施顺序

1. Core 实现三类内部值对象、strict reader/writer、identity 与 contract fake。
2. Core 实现纯 comparator，覆盖 mandatory/preference、五类比较、证据和 stale fail-closed。
3. Compiler 添加现有 `_compiler.TargetCapabilities` 的保真 adapter，不改旧 fingerprint/comparator。
4. Platform 先接 CPU adapter，再用第二个 fake/remote producer 做替换 conformance。
5. Runtime 接 matching seam、独立 CPU candidate、六轴 fallback 和 decision/evidence 关联。
6. 各延后域逐项提交 contract proposal；真实硬件能力另走认证路径。

实现依据为 `flagquantum/core/target_capabilities.py` 及 Core、Runtime、Compiler、
Provider 的能力匹配与替换测试。历史实现授权记录不再作为代码契约。

## Phase 2 集成收口状态（2026-09-04）

上述顺序的内部最小闭环已在集成提交 `0f63c3b4` 完成：Core contract/matcher、Compiler adapter、
CPU Platform producer、Runtime policy seam，以及 synthetic Execution 第二 producer replacement
conformance 均已合入并通过总控定向验证。Runtime 合入后为 `121 passed`，第二 producer 修复后为
`118 passed`；两次均通过 architecture、team scope、Ruff 与 diff check。

这里的“完成”只表示本裁决批准的内部边界闭环完成。Runtime seam 不在默认执行路径，内部 decision
record 不是稳定 Execution Request/Result/Evidence；synthetic producer 不访问网络、provider SDK 或
真实硬件。公共 API、默认 backend/fallback、plan/result、actual execution evidence、真实 provider/
hardware 和所有延后域仍未授权。

下一阶段应先偿还 Runtime candidate provenance/非 CPU fallback 可信绑定、Compiler legacy comparator
共判接线和剩余确定性/CPU full-rematch 测试债务；之后再单独评审 execution observation/evidence
proposal 与真实国产 Platform adapter 的认证前置工作。本状态记录不自动启动这些任务，也不修改本
裁决的机器授权边界。
