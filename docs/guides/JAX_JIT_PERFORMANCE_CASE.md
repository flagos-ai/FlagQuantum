# JAX JIT 性能案例设计

## 要回答的问题

同一套 FlagQuantum Statevector 损失与梯度任务，在保持 PyTorch 训练接口的前提下：

1. JAX JIT 第一次调用需要付出多少编译成本？
2. 编译完成后，JAX 单步相对 PyTorch Native 快多少？
3. 随 qubit 数增加，首次编译成本如何变化？
4. 重复调用多少次后，JAX 的累计时间开始低于 PyTorch？

## 冻结实验协议

- 路径语义：单张 A800 的 `single_device_fast_path`，不作分布式扩展声明。
- 方法：精确 Statevector，complex64。
- 电路：每层 RX/RY/RZ、线性 CX、首尾 RXX。
- 指标：loss + backward gradient 的端到端时间。
- 冷启动：每个 qubit 规模在新 Python 子进程执行，并禁用持久化编译缓存。
- 稳态：第一次 JAX 调用完成后，重复执行并取所有冷启动进程样本的中位数。
- 正确性：JAX/PyTorch loss 与梯度最大绝对误差均不超过 `1e-4`。
- 扫描：默认 4/6/8/10/12 qubits、2 layers、3 次冷启动、每次 10 个稳态样本。

## 图片规划

| 图 | 核心问题 | 主要指标 | 推荐结论 |
| --- | --- | --- | --- |
| 01 Cold Start | 首次编译有多贵？ | PyTorch first vs JAX build+first | 大规模输入放大 JAX 首次编译等待 |
| 02 Steady State | 编译后是否值得？ | loss+gradient 稳态延迟 | 小规模高频调用可受益于 JIT 融合 |
| 03 Speedup | 稳态收益多大？ | PyTorch/JAX 加速比 | 只陈述冻结工作负载的开发结果 |
| 04 Break-even | 多少步后回本？ | 累计时间交叉所需调用数 | Planner 应结合复用次数选择后端 |

展示时必须同时注明 CPU/依赖版本、首次编译是否计入、warm/cold 定义、重复次数、
正确性误差和 `single_device_fast_path`。不得把结果描述成多卡扩展性证据或所有规模
上的通用后端结论。

## 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/jax_jit_crossover.py \
  --wires 4,6,8,10,12 --layers 2 \
  --cold-repetitions 3 --steady-repetitions 10 \
  --device cuda:0 \
  --output benchmarks/results/local/jax_jit_crossover_a800.json
```

Plotting and publication-specific analysis are maintained outside the source
repository; the JSON result is the reproducible interface.

## A800 实测结果

环境：单张 NVIDIA A800-SXM4-80GB，PyTorch 2.13.0+cu130，JAX 0.10.2；
每个规模使用 3 个全新进程测量冷启动，每个进程采集 10 个稳态样本。

| Qubits | JAX 冷启动 | PyTorch 稳态 | JAX 稳态 | JAX 稳态加速 | 回本调用数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5.015 s | 26.589 ms | 0.958 ms | 27.75× | 190 |
| 8 | 8.761 s | 43.124 ms | 2.195 ms | 19.65× | 210 |
| 12 | 12.662 s | 60.018 ms | 3.356 ms | 17.88× | 221 |
| 16 | 14.480 s | 75.256 ms | 10.498 ms | 7.17× | 221 |
| 20 | 17.100 s | 132.453 ms | 127.120 ms | 1.04× | 3147 |

所有规模的 loss/gradient 正确性均通过冻结的 `1e-4` 阈值。结果支持的核心结论是：
JAX JIT 在重复的小规模模拟中具有显著稳态优势，但首次编译成本随规模增加；到
20 qubits 时稳态优势几乎消失，需要约 3147 次同形状调用才能摊平冷启动成本。
