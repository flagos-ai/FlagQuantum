# 多深度横场 Ising VQE：JAX JIT 收益区间与编译回本边界

## 端到端训练轨迹（Adam 500 steps）

单步 benchmark 之外，`benchmarks/jax_jit_vqe_training.py` 使用相同初始化、Hamiltonian、
Ansatz、Adam 超参数和 500 个真实 optimizer steps，分别执行 PyTorch native 与 JAX JIT。
JAX 墙钟时间包含 kernel 构建和首次编译；每一步包含 forward、backward、optimizer update
与设备同步。精确基态能量由 SciPy sparse `eigsh` 计算，收敛阈值为能量误差 `1e-3`。

| Qubits | Depth | 500 步内收敛 | 首次达标步数 | JAX 累计回本步数 |
| ---: | ---: | :---: | ---: | ---: |
| 4 | 2 | 是 | 175 | 136 |
| 4 | 4 | 是 | 92 | 157 |
| 4 | 8 | 是 | 72 | 141 |
| 8 | 2 | 否 | — | 145 |
| 8 | 4 | 否 | — | 131 |
| 8 | 8 | 是 | 110 | 122 |
| 12 | 2 | 否 | — | 147 |
| 12 | 4 | 否 | — | 108 |
| 12 | 8 | 否 | — | 120 |

JAX 在所有配置中最终都完成累计耗时回本，但“性能回本”不等于“算法收敛”。8 qubit
浅层和全部 12 qubit 配置，在当前 Ansatz、初始化、Adam 学习率与 500 步预算下均未达到
`1e-3`。PyTorch/JAX 的首次达标步数一致，说明 JAX 主要改变求解速度，没有改变按步优化
行为。

L-BFGS 路径已实现并记录 closure evaluation 次数；本轮未获得 A800 执行授权，因此没有把
L-BFGS 标为实测证据。

## 案例目标

用真实的 VQE energy+gradient 工作负载回答：JAX JIT 在什么 qubit 规模和电路深度
下值得启用，首次编译需要多久，重复多少步后能够回本。

## 冻结任务

- Hamiltonian：`H = -0.7 Σ ZiZi - 0.25 Σ Xi`，开放边界。
- Ansatz：每层 RX/RY/RZ、最近邻 CX、首尾 RXX。
- 模拟：精确 Statevector、complex64。
- 训练计算：Hamiltonian energy + 全参数 backward gradient。
- 网格：qubits `{4, 8, 12, 16, 20}` × depth `{2, 4, 8}`。
- 后端：FlagQuantum PyTorch Native 与 PyTorch 接口下的 JAX JIT 量子内核。
- 冷启动：每点使用全新 Python 进程并关闭持久化编译缓存。
- 语义：`single_device_fast_path`，不构成分布式扩展性证据。

## A800 实测结果

环境：单张 NVIDIA A800-SXM4-80GB，PyTorch 2.13.0+cu130，JAX 0.10.2。

| Qubits | Depth | JAX冷启动 | PyTorch稳态 | JAX稳态 | 加速比 | 回本步数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 / 4 / 8 | 4.5 / 8.3 / 13.7 s | 26.7 / 46.7 / 82.6 ms | 1.0 / 1.6 / 2.8 ms | 26.9 / 29.2 / 29.4× | 171 / 182 / 171 |
| 8 | 2 / 4 / 8 | 8.2 / 13.4 / 22.8 s | 45.0 / 87.0 / 164.3 ms | 1.9 / 3.2 / 6.3 ms | 24.2 / 27.1 / 26.3× | 190 / 159 / 145 |
| 12 | 2 / 4 / 8 | 12.2 / 15.9 / 31.9 s | 66.9 / 127.0 / 238.9 ms | 3.0 / 5.5 / 10.7 ms | 22.5 / 23.1 / 22.4× | 190 / 131 / 140 |
| 16 | 2 / 4 / 8 | 14.1 / 22.9 / 44.9 s | 99.2 / 170.3 / 325.0 ms | 9.2 / 16.9 / 27.0 ms | 10.8 / 10.1 / 12.1× | 156 / 149 / 151 |
| 20 | 2 / 4 / 8 | 17.7 / 28.4 / 50.6 s | 126.6 / 216.3 / 417.5 ms | 93.9 / 163.6 / 318.0 ms | 1.35 / 1.32 / 1.31× | 534 / 533 / 505 |

全部 15 个点均通过 loss/gradient `1e-4` 正确性阈值。

## 结论

1. 4–12 qubits 是明显的 JAX JIT 收益区，稳态加速约 22–29×。
2. 16 qubits 仍有约 10–12× 收益，但首次编译可达 44.9 秒。
3. 20 qubits 时稳态收益缩小到约 1.3×，冷启动最高 50.6 秒，需要约 505–534
   次同形状 energy+gradient 调用才能回本。
4. Planner 不应仅依据 qubit 数选择 JAX；还要结合 depth、预计训练步数、shape
   稳定性和编译缓存命中情况。

## 图片叙事

1. `01_vqe_jax_benefit_map.svg`：稳态加速收益区间。
2. `02_vqe_compile_cost_map.svg`：首次编译成本随规模和深度增长。
3. `03_vqe_break_even_map.svg`：累计调用回本边界。
4. `04_vqe_backend_decision_map.svg`：将收益和编译成本合成为后端选择建议。

这些结果是单次冷启动开发测量，适合案例展示和 Planner 假设形成；正式发布声明应
增加至少三次独立冷启动、置信区间和第二台机器复测。
