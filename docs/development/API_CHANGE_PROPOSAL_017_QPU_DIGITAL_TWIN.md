# API Change Proposal 017：QPU数字孪生领域入口

## 状态

**Approved before the first public alpha.** API owner 于2026-09-09明确授权将
已有数字孪生研究成果按vNext边界加入 `flagquantum/twin`。仓库尚未公开发布，
不保留旧研究模块名称或兼容入口。

## 决策

- 新增候选公共命名空间 `flagquantum.twin`，不增加根命名空间导出；
- 首批入口为 `QPUDigitalTwin`、`TwinSnapshot`、`TwinPrediction` 和
  `TwinValidationReport`；硬件验证入口增加 `TwinExperiment` 和
  `TwinHardwareReport`，不引入重复的任务轮询或Run Manager；
- Twin拥有校准条件化设备模型、冻结预测和真机比较；
- Noise拥有噪声语义，Simulation拥有数值执行，Remote拥有厂商校准获取与任务控制；
- Q-ATLAS、阶段编号、候选状态机和实验数据流程不进入正式公共命名空间。

## 成熟度边界

首个纵向切片只承诺单线路、精确密度矩阵预测以及基于测量计数的分布比较。
它是校准驱动的设备仿真与验证接口，不代表脉冲级等价，也不自动形成跨设备、
跨校准周期的预测性数字孪生声明。

## 验收

- Quafu校准可以经现有Remote转换器冻结为孪生快照；
- 快照绑定设备画像、噪声模型、物理映射和采集时间；
- 预测复用现有Noise与Simulation执行链路；
- 真机计数比较产生可序列化、身份绑定的验证报告；
- 提交线路在发送前冻结，提交回执和结果必须携带相同摘要；提交身份不等同于
  执行身份。公开Quafu任务接口未提供提交前最终线路回执；本地QSteed或
  QuarkCircuit转译结果只能作为候选线路，不能替代平台回执。因此当前真机报告
  固定标记为事后诊断验证，并另行记录平台事后返回的执行线路是否与提交线路一致；
- `twin` 不被 Simulation、Noise 或 Remote 反向导入。
