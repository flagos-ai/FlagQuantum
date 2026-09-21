# CPU statevector performance characteristics

Status: local development measurements, recorded. Not a support claim, not a
release gate, and not a cross-framework comparison.

This document records what the CPU statevector path costs, what a series of
changes bought, and which plausible-looking optimizations were measured and
rejected. The rejected entries are the point of the exercise: each one is a
tempting change that either does nothing or makes the path slower, and without a
record the next reader pays for the experiment again.

Every number below was produced on the host described in
[Measurement scope](#1-measurement-scope) and means nothing off it. Where a
ratio is quoted, it is a paired comparison of two arms measured in one process
with the arms alternating, because this host cannot resolve a difference between
two separately-measured runs. Per `docs/guides/PERFORMANCE_ENGINEERING.md`, these
are engineering measurements, not the reviewable artifacts a published
performance claim requires.

## 1. Measurement scope

- macOS 27.0 arm64, Python 3.12.14, Torch 2.13.0, no CUDA device.
- `os.cpu_count() == 10`; Torch's default thread count on this host is 4.
- Complex64 statevectors, 20 wires unless a row says otherwise, which is
  `2**20 * 8 B = 8 MiB` of state.
- Warmup before timing; median of several rounds; the arms of a paired
  comparison alternate round order.
- Local CPU only. Nothing here describes a GPU, a distributed run, or a QPU.

Two conventions matter for reading the rest:

- **A paired ratio is the measurement.** A ratio taken between two arms measured
  in separate runs or in separate processes is not reported, because on this host
  it is dominated by drift.
- **"Held-out" means the pre-change behaviour**, usually reconstructed by
  defeating the new branch in both modules so the old kernel is genuinely
  reached. It is an emulation, and the per-kernel call counts recorded beside it
  are what show the emulation worked.

## 2. The reference the numbers are read against

| Quantity | Measured | Meaning |
| --- | ---: | --- |
| `torch` copy bandwidth, 64 MiB tensor | 1.309 ms, **102.6 GB/s** | the practical ceiling for a pass over the state |
| Shipped single CX (`_apply_cx_permutation`) | 0.1887 ms | **88.9 GB/s**, 87% of that ceiling |
| `torch.bmm` for a one-wire gate, 20 wires | 3.148 ms | the dense single-qubit kernel |
| Same gate as reshape to 4-D plus `stack` | 0.809 ms | 3.89x faster, and the reason the elementwise kernels exist |
| H+160 RZ end to end | 58.04 ms → 27.72 ms | 2.09x once the one-wire kernels stop permuting |
| 312-gate depth-8 `state(refresh=True)` | 680.50 ms | **11.54 GB/s**, about 11% of the copy ceiling |

The last row is the honest summary of the whole exercise: a real circuit spends
most of its time in per-gate passes over the state rather than in any single
slow kernel, so the available wins are in reducing the number of passes, not in
making one pass faster. The single CX is already at 87% of what the memory system
will give, which is why the CX kernel is not the place to look.

## 3. What this series changed

Ratios are the paired, interleaved measurement. The last column says what the
change is, because two of these are not speed changes and saying otherwise would
be the easiest way to mislead a reader.

| Change | Paired ratio | Kind of change |
| --- | ---: | --- |
| Fused CX sequence applied with one gather | 145x cached at length 152; 1.78x at length 2; **0.85x at length 1** | speed, above a length threshold |
| One-wire gate without the layout permutation | `rotation_chain` 3.33x, `diagonal_chain` 3.24x, `mixed_chain` 2.88x | speed |
| Diagonal fused region routed to the diagonal kernel | `diagonal_chain` 2.03x, `two_wire_diagonal_chain` 1.50x, `rotation_chain` 1.00x, `mixed_chain` **0.94x** | speed, and a measured regression on one case |
| Joint marginal read off the dense state's own distribution | reduction alone 30.4x at k=2 to 4646x at k=8; public entry point **1.11x to 1.18x** | speed, but no longer the bottleneck |
| Z expectation read off a bounded marginal, sign table capped | memory **79.7 MB resident across 19 keys → 0**, table ceiling 1 GiB | **capacity**, and only a narrow-request speed win |
| Vectorized trajectory row sampling | 5.2x to 6.2x isolated; 1.49x to 4.44x end to end | speed, opt-in, **different seed mapping** |

Notes that belong beside those rows:

- **The gather's crossover is real and is handled.** Below a sequence length of
  about 4 the index build is not amortized and the gather loses; the classifier
  only takes the gather path where the sequence is long enough to pay for it.
  Length 1 at 0.85x is why.
- **The diagonal classifier regressed `mixed_chain` to 0.94x** while the
  elementwise arm on the same case reached 2.63x. The classifier is a
  two-way choice made per fused region and it does not always pick the winner;
  the case is recorded rather than smoothed over.
- **The joint-marginal public ratio near 1 is the honest framing.** The reduction
  itself became 30x to 4646x faster, and the caller sees 1.11x to 1.18x, because
  on a 20-wire program the reduction was never what the caller was paying for.
- **The Z-expectation change is a memory fix.** `expectation_z` on 20 wires with
  all wires requested built a `(2**20, 20)` float32 sign table, 10x the state,
  and the per-wire request pattern retained 19 keys with no eviction. Nothing
  about the arithmetic changed. The only wall-clock gain that survives is for a
  narrow request: against a warm table the marginal wins up to k = 16 (0.029x to
  0.315x of the table's time, best at k = 2) and loses from the high teens on,
  1.52x at k = 19 and 1.87x at k = 20, which is why the marginal arm is bounded
  at 8 wires, a comfortable interior point rather than a cliff edge.

### 3.1 The row-sampling number the plan got wrong

An earlier planning document quoted **41.5x** for vectorizing trajectory branch
sampling. That figure is not reproducible against the shipped contract and should
not be quoted. The probe it came from drew **one** uniform vector from **one**
generator for every trajectory, which is not what the function does: each
trajectory owns the generator its id derives, and the runtime documents that the
trajectory batch size does not change a seeded result.

Priced with the uniform drawn per row from that row's own generator - the shape
that keeps the contract - at 4096 trajectories x 4 branches:

| Arm | Per sampler call | Ratio |
| --- | ---: | ---: |
| Shipped per-row `multinomial` loop | 21.758 ms | 1.0x |
| Inverse CDF, per-row generators (contract-preserving) | 4.206 ms | **5.17x** |
| Inverse CDF, one generator for all trajectories | 0.051 ms | 604x |
| The per-trajectory `torch.rand` calls alone | 2.421 ms | — |
| The batched `cumsum` + `searchsorted` search alone | 0.028 ms | — |

The 604x arm is not a faster version of the shipped function; it is a different
function. The gap between 5.17x and 604x is the per-trajectory uniform draws, not
the search, which is why the search cannot be optimized into the plan's number.

End to end, the sampler is 41% to 95% of a trajectory run depending on the
circuit, and the paired ratios follow:

| Case (4096 trajectories, 6-16 layers) | Per-row | Vectorized | Ratio | Sampler share |
| --- | ---: | ---: | ---: | ---: |
| 2-wire `bit_flip`, shallow | 215.5 ms | 69.7 ms | 3.09x | 77% |
| 2-wire `bit_flip`, deep | 778.0 ms | 175.4 ms | 4.44x | 95% |
| 8-wire `bit_flip` | 3562.6 ms | 1960.6 ms | 1.82x | 54% |
| 8-wire `depolarizing` | 4524.6 ms | 3033.5 ms | 1.49x | 41% |
| 2-wire `bit_flip`, 16384 trajectories | 1573.9 ms | 429.4 ms | 3.67x | 92% |

The win is real where the sampler dominates - shallow circuits on few wires - and
diluted to 1.49x where the statevector work sits beside it. The switch is opt-in
because a uniform inverted against a cumulative sum and `torch.multinomial` are
different estimators over the same distribution: the same seed does not produce
the same branch, and seeded trajectory runs are replayed.

## 4. Rejected approaches

Each of these was tried and measured. None of them is worth revisiting without a
new reason.

| Do not | Basis (measured on this host) |
| --- | --- |
| Move the Triton kernels to CPU, or expect `FQ_TRITON_*` to do anything on CPU | Every Triton branch is gated on `state.is_cuda` and `dtype == complex64` (`flagquantum/simulation/statevector/local.py`, `_apply_cx_sequence`, the one-wire matrix regions, the `ry`/`rz` pair, and the loop switch). With no CUDA device those are dead code, and the switches are CPU no-ops rather than CPU kernels. |
| Use `torch.compile` as CPU kernel fusion | One-wire eager elementwise 0.971 ms vs the same code compiled **2.387 ms**, 2.46x slower; compilation 11.64 s; inductor emits `Torchinductor does not support code generation for complex operators. Performance may be worse than eager.` Compiling the `bmm` form came to 0.974x, i.e. no gain either. |
| Tune threads or `OMP_NUM_THREADS` | 1/2/4/8 threads on the same 20-wire circuit: 252.2/248.6/242.2/222.9 ms, best case **1.13x**, and Torch already defaults to 4 here. The loop is bandwidth-bound, so extra threads do not buy bandwidth. |
| Rewrite the single CX permutation kernel | Shipped 0.1887 ms is 88.9 GB/s, 87% of the 102.6 GB/s copy ceiling. The alternatives are worse: `index_select` **15.2x slower** (2.862 ms), and moving the target axis innermost plus a flip **2.30x slower** (0.434 ms). Both are bitwise exact, so this is not a correctness-versus-speed trade; there is simply no headroom. |
| Replace `expectation_z`'s reduction with a hand-written reshape-sum | Shipped 2.377 ms is **faster** than the reshape-sum form at 3.043 ms (1.28x). The existing path is not a defect. |
| Rewrite *full-state* shot sampling with a cumulative-sum search | `multinomial` 1.712 ms vs cached CDF `searchsorted` 1.583 ms, only **1.08x**. This is the sampling path that builds one distribution over `2**n` outcomes and draws 8192 shots; it is not the trajectory row sampler of §3.1, where the win is large because the cost is one `multinomial` dispatch per (trajectory, event) rather than one search over a full distribution. Do not read the §3.1 result as contradicting this row. |
| Reuse one generator across trajectories to batch the draw | It is faster by two orders of magnitude and it breaks the documented per-trajectory stream: the trajectory batch size would change a seeded result. This is the shape that produced the disowned 41.5x. |
| Cache the sign table instead of bounding it | At 20 wires a `(2**20, 20)` table is 80 MB, 10.0x the state, and the per-wire request pattern retains 19 keys without eviction, 76 MiB resident. Widening to n = k = 30 extrapolates to roughly 129 GB against an 8.6 GB state. Pre-allocating the int64 fill is only 1.33x and the float32 variant is slower, so the fix is a bound plus a marginal, not a better table. The cache surviving `_invalidate_execution_cache` is deliberate and is not part of this. |

## 5. Registered adjacent subsystem

`flagquantum/simulation/statevector/split_real_imag.py`, exposed through
`flagquantum/experimental/numerics.py`, already implements a real/imaginary split
statevector. It is **not** a CPU acceleration path and this series did not build
on it, and registering that here is the point of the entry: it looks like a
ready-made answer and it is not one.

- Its apply path, `instruction_matrix_pair` / `apply_gate_pair`, uses the same
  `reshape` -> `permute` -> `matmul` -> `permute` back shape as `_apply_matrix`,
  so it takes the same passes over the state. Splitting a complex tensor into two
  float tensors does not reduce traffic.
- Its gate set is `SPLIT_REAL_IMAG_SUPPORTED_GATES`, a subset of the gates the
  main path handles.
- The gather-based kernels in this series were measured against a split float32
  arm as well, and the split arm was not the winner.

A future change may well want a split representation, for register pressure or
for numerical reasons. It is not free statevector bandwidth today.

## 6. Reproducing these numbers

The probes are development scripts, not committed benchmarks, and they are not
part of CI. The measurements above are recorded here so the conclusion survives
without them; anyone re-running them should expect to reproduce the shape of the
result, not the exact ratios, on the same host.

Before trusting any new measurement on this machine:

1. Establish the copy bandwidth of the day. Every kernel ratio is only
   interpretable next to the ceiling on the same run.
2. Interleave the arms in one process and alternate round order. Two runs
   measured separately will disagree by more than the effect.
3. Report the per-kernel call counts of the held-out arm. Without them, "the old
   kernel was reached" is an assumption.
4. State which switch state each arm ran in. Three CPU switches exist
   (`FQ_CPU_CX_SEQUENCE_GATHER`, on by default and bitwise exact;
   `FQ_CPU_SINGLE_WIRE_ELEMENTWISE` and `FQ_CPU_VECTORIZED_ROW_SAMPLING`, opt-in
   because their output is not bitwise the output of what they replace), and a
   ratio measured with the wrong one set is a ratio for a different program.
