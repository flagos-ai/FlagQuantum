# IR Phase 2 Batch F 实现复核

## 结论

Batch F 已在私有、离线、显式调用边界内形成端到端静态编译闭环：

```text
CircuitIR import/seal
  -> static canonicalization
  -> RX/RY/RZ/CX target decomposition
  -> directed placement/routing
  -> post-routing canonicalization
  -> deterministic identity/cache
  -> OpenQASM 2 / OpenQASM 3 / QCIS text emission
```

该闭环不发现云资源、不接收真实 backend ID、不读取凭据、不调用 provider SDK、
不提交任务，也没有进入公开 API 或默认执行路径。

## 已形成的证据

- 版本化离线 corpus 覆盖 legacy-native、合成 Quafu-static 线性路由和定向反转；
- 输出门集结构、定向耦合合法性和输出哈希由测试锁定；
- OpenQASM 2、OpenQASM 3、QCIS 均回解析并验证状态与 qubit order；
- 与 legacy QASM/QCIS 发射结果进行解析后语义差分；
- 无文本发射模式验证 expectation 与 autograd gradient；
- 编译身份、校准快照变化、缓存 miss/hit 和字符输出确定性均有机器证据；
- measurement、observable、dynamic circuit、unbound parameter、noise channel、
  qubit-count mismatch 均 fail closed，避免静默丢失部署语义。

## 性能基线

在 `flagquantum-dev:pr-check`、单线程 CPU、5 次采样下，完整链路 10K gate 的
cold/cached p95 分别为 574.794476 ms / 440.719608 ms，cold peak host memory
为 26,196,356 bytes。该结果只是私有工程基线，不是公开 SLA。

性能预算已经形成候选，但仍未获 owner 批准，因此尚未成为回归门禁。

## 仍然关闭的边界

- adapter 源码、provider SDK 与远程提交；
- token、凭据、真实 backend ID、队列、任务与价格；
- 公开 API、默认路径切换和 legacy retirement；
- Phase 2 自动退出。

## 下一授权点

若认可当前基线与候选阈值，使用精确命令：

`approve IR-PHASE2-BATCH-F-PERFORMANCE-BUDGET`

批准后才可把候选预算固化为正式门禁，并继续准备 Phase 2 exit review candidate。
