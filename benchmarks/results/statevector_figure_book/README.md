# FlagQuantum 分布式态矢量证据图册

本图册基于 NVIDIA A800 实测 development evidence。它不依赖 Git commit、
GitHub 或 release 签名；也不把 development evidence 描述成正式发布认证。
cuStateVec 仅用于隔离环境中的单卡 forward 黑盒比较，不是 FlagQuantum
运行时依赖。

## 阅读顺序

1. `fig01_same_host_strong_scaling`
   回答固定 28q 可微训练在同机 1/2/4/8 卡上的时间、加速比、并行效率、
   每 rank 显存和梯度正确性。8 卡端到端加速 6.70x，backward 加速
   6.17x。
2. `fig02_compute_communication_topology_ablations`
   解释加速来自哪里：融合 VJP、跨分片 CX 半空间打包，以及多节点
   topology-aware rank-bit 布局。三项分别作用于算子、通信量和跨节点路由。
3. `fig03_circuit_family_generality`
   检查优化是否只适用于一种电路。Full-width Linear、Ring 和 Brickwork
   都获得端到端收益，并保持 raw gradient 误差远低于 complex64 的
   3e-5 阈值。
4. `fig04_science_training_and_capacity`
   给出端到端结论：30q 科学训练由单卡 23.45 s 降到两节点 16 卡
   4.32 s；35q 在单张 80 GB A800 上发生 OOM，而 16 卡完成训练。
5. `fig05_external_custatevec_baseline`
   给出当前单卡 kernel 水平：同一张 A800、同一 20q complex64
   RY/RZ/CX 精确电路下，FlagQuantum 为 2.185 ms，cuStateVec 为
   1.259 ms，差距 1.74x；最终态最大误差 1.33e-7。
6. `fig06_statistical_training_speed`
   展示 frozen speed sweep 中的 24q、8 层、376 门、192 参数训练点。
   1 卡和 8 卡均使用 5 次预热、30 次同步测量；8 卡端到端加速
   1.644x，bootstrap 95% CI 为 [1.636, 1.653]。图中同时保留
   Adam 与参数广播的 6.47x 回退，明确展示当前深电路瓶颈。
7. `fig07_parameter_synchronization`
   展示 owner-sharded Adam 的同步优化：192 次逐参数 broadcast 合并为
   一次 owner-masked packed all-reduce。optimizer+sync 从 15.35 ms
   降至 9.67 ms，提升 1.59x；更新后参数最大差异为 0。

## 证据边界

- 1/2/4/8 卡图为同机 A800 强扩展；16 卡图为两节点 A800。
- 外部比较是单卡 forward，不代表分布式训练能力对比。
- 35q 的 256 GiB 是失败分配请求，不是 35q 训练的稳定常驻显存。
- 当前图册证明实测能力和优化效果，不等同于 ISSUE-044 的签名 release。
- 图中的结论只覆盖 complex64、所列 gate set、workload 和拓扑。

每张图均提供 PNG、SVG 和 PDF；合并版为
`FlagQuantum_Distributed_Statevector_Figure_Book.pdf`。
