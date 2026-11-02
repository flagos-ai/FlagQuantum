# Complete-circuit precision evidence for issue #901

This development comparison characterizes existing FP32/P1, selective P2,
host-encoded Double-Single/P3, device-generated Double-Single/P4 and native
FP64 Circuit routes. It does not introduce a precision abstraction, production
API, mandatory vendor dependency or capability-maturity promotion.

## Inventory, protocols and primary evidence

The producer reuses the staged numerical implementations, operator preflight
and conformance runners, the common benchmark metadata/writer contract, and
the gradient/precision/speed tutorial. Scalar arithmetic conformance and the
older staged A800 records remain useful scope evidence; neither is used as a
complete-circuit speedup measurement here. Protected contracts and profiles
remain unchanged.

The frozen protocol is
`benchmarks/manifests/precision_frontier_v1.json`. Workloads use widths/depths
(2,2), (2,8), (4,8), seeds 11/29, interference and noncommuting gates, and a
cancellation echo family with ordered large/small/opposite Pauli coefficients.
There is one named RY occurrence and a supported parameter-shift derivative.
All routes consume identical FP32-representable initial angles, dyadic fixed
angles and exactly represented coefficients. No candidate receives a more
accurate initialized parameter than its comparator.

- [CPU report](cpu/REPORT.md): 60 completed cases, including P2/P3 diagnostic
  controls, with five repeated samples and no discarded inaccurate cases.
- [CUDA report](cuda/REPORT.md): 60 completed cases on the repository's
  NVIDIA A800-SXM4-80GB runner, using the same frozen inputs and five samples.
- [Apple MPS report](mps/REPORT.md): 36 explicit unsupported full-circuit
  combinations, refused after preflight without launching substitute workers.
  The existing Double-Single arithmetic conformance itself passed on real
  Apple MPS with host fallback disabled. P0--P4 and P5 optimizer do not have
  a registered MPS platform in this source. P5 autograd is CPU-only; native
  FP64/complex128 is unavailable on MPS. These are actual stage boundaries,
  not new declarations of support or missing data scored as fast execution.

The producing CPU/CUDA/MPS source is
`3ba42c605f36d86ce52c499dc216fab72f4ddcac`, with `dirty=false` in every freeze
and case. The actual local device is Apple M5; a read-only hardware observation
and inherited compiler-related flags are retained under `validation/`.
Hostnames and local paths are canonically redacted for publication. Source
identities, numeric values, targets and raw timing samples are preserved.

## Independent numerical checks

The reference constructs NumPy complex128 matrices directly, enumerates the
big-endian embedding and propagates an analytic initial-RY tangent. It never
calls the candidate simulator or its gate generators. Tests also compare that
tangent and the energy derivative with centered finite differences. Repeated
Z coefficients are coalesced using `math.fsum` before reference evaluation,
so the reference evaluates the physical operator without reproducing the
candidate's reduction cancellation.

Each case records amplitude maximum/L2 error, fidelity/infidelity, norm drift,
native Z0/X1/Z0Z1/objective reductions and the supported parameter gradient.
Separate fields retain the actual stage gate-synthesis diagnostic, coefficient
conversion error, initial-angle conversion error, and its reference-state
consequence. `quality_vs_intended_inputs` also reports the counterfactual
error against the unrounded intended FP64 angle.

The frontier's declared tolerance applies to the actual matched input circuit,
not that counterfactual unrounded input. Input conversion can exceed a target
even when the matched-input frontier is feasible. Gate-synthesis diagnostics
use the actual staged encoders; the FP64 diagnostic uses the existing device
GATE_MAT_DICT generators and does not inspect constants inside fused native
kernels. It must not be presented as a complete vendor-internal route audit.

Independent small arithmetic tests use exact Decimal references for a
residual-bearing product and catastrophic cancellation. All returned candidate
state/observable/gradient words are checked for actual requested device and
dtype. Native FP64 uses Float64 words; all split/DS words remain Float32.
A CPU-returning or dtype-demoting candidate is rejected rather than reclassified.

## Timing, peak memory and bit identity

Every case has a fresh process and a 240-second deadline. Repeated complete
API calls include circuit construction, gate generation, state evolution,
observables, parameter-shift gradients and result metadata. Device
synchronization brackets every absolute sample. Five samples, a correctly
named median, first call including preflight, warmup and process wall all
remain available. The optional GPU job has a separate bounded watchdog.
No GEMM ratio is called a simulator speedup.

Peak worker RSS at timing end includes imports, preflight, warmup and output
hash transfers. CUDA allocator peak covers the repeated samples and resident
outputs; it is not total exclusive device memory or an audit of unobserved
allocators. Exact state storage is separate: two FP32 words for P1/P2, four
FP32 words for P3/P4, and complex128 storage for native FP64. The frontier
uses measured peak memory (CPU RSS or CUDA allocator peak), never state
storage as a substitute. Unknown accelerator peaks cannot win a Pareto score.
No CPU/GPU peak-memory or timing ranking across different devices is made.

Hashes include actual dtype, shape and bytes of every retained repeated
state, observable and gradient word. A repeated bit-identity statement is
within one route/device and based on these hashes. Scalar-error equality,
fidelity equality and rounded printed values do not establish identity.
A regression test deliberately gives equal scalar errors with different
output hashes, including the distinct encodings of positive and negative zero.

All slower, inaccurate, unsupported, failed and timeout cases remain visible.
An infeasible tolerance yields no feasible mode, not an infinite speedup.

## Observed limits and interpretation

On CPU, full P4 sharply reduces state error on the noncommuting cases but is
much slower than native FP64 in this bounded eager implementation. The large
coefficient cancellation cases show why excellent state accuracy alone is
insufficient: observable and gradient accumulation can still miss a tighter
objective tolerance. P2 improves reductions without changing FP32 state
storage/evolution. P3's host-generated gates are a distinct cost/domain from
P4's device-generated gates.

Double-Single retains binary32 exponent range and is not IEEE binary64.
Dekker splitting requires finite values below the overflow range of
multiplication by 4097; intermediate products, subnormals and flush-to-zero
behavior further limit useful precision. Roughly 48 significant bits is not
a universal error bound. P4 direct scalar angles are bounded by 1024, with
its existing fixed/RX/RY/RZ/RXX/RYY/RZZ gates. Arbitrary expressions, general
native autograd, distributed execution and convergence are not certified.
P5's separately probed delivery/optimizer interfaces do not extend this
parameter-shift benchmark's domain or promote maturity.

The direct native Circuit tensor API exposes output residency/dtype but does
not supply a gate/parameter host-route trace. The producing snapshot's fixed
native `host_gate_encoding`/`parameter_host_fallback` labels must be treated
as unknown boundaries, not vendor-route evidence. The current producer makes
that limitation explicit. P3's declared CPU gate encoding remains known and
is not disguised as a device-only route. Returned tensors on the requested
device and absence of a provider-internal route audit are separate statements.

## Reproduction and source audit

Use the existing optional numerical comparison environment; actual resolved
Torch/NumPy/build/runtime versions accompany every worker. From the producing
checkout, use new output directories:

```bash
python -m benchmarks.precision_frontier --smoke --output /tmp/precision-smoke
python -m benchmarks.precision_frontier --staged-controls --output /tmp/precision-cpu
python -m benchmarks.precision_frontier --device cuda:0 --staged-controls --output /tmp/precision-cuda
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m benchmarks.precision_frontier --device mps --output /tmp/precision-mps
python -m benchmarks.precision_frontier --report --output /tmp/precision-cpu
python -m pytest tests/benchmark_contract/test_precision_frontier.py -q
python artifacts/development/precision_frontier_cpu_cuda_mps_20261102/validation/verify_sources.py
```

The common Local GPU Gate's manual `precision_frontier=true` input uses the
existing A800 lane and source preparation, exclusive device concurrency,
watchdog and raw artifact upload. Select only one manual campaign input.
Smoke is an execution check, not full acceptance evidence.

`validation/verify_sources.py` resolves recorded producer/manifest bytes from
Git, compares their SHA-256 values, requires the full frozen case inventory,
and checks raw samples/statistics, output hashes, actual dtype/device guards
and peak-memory/storage scopes. Report regeneration validates the same source
and protocol and refuses missing repeats or incorrect statistics.

The initial producer
`85de27031547f0819c97c6fe681eb4b63587efd2` ran the same numerical inputs and
protocol before output-word guards and the physical MPS preflight distinction
were added. Its entire CPU/CUDA/MPS discovery JSON is retained in the
SHA-256-bound `validation/initial-discovery.json.gz` archive, separately from
primary results. The initial MPS preflight incorrectly required a registered
runtime even for generic Torch arithmetic and classified its native FP64
refusal as failed; both behaviors are corrected in the final producer.
No numerical target or circuit was changed to select a better outcome.

Public logs are losslessly gzip-compressed after the existing infrastructure
sanitizer. The archive and report source revisions remain published and must
be preserved by a merge commit rather than squashed away.

## Accelerator receipt and acceptance assessment

The final [Local GPU Gate run](https://github.com/flagos-ai/FlagQuantum/actions/runs/41401959037)
passed on the pinned source. CUDA runtime 13.0 and physical device UUID
`dd0e4377-e2a5-f437-cdd8-d9a7a16e18c9` accompany the worker results. The
producer could not query the NVIDIA driver version; its null value is an
observability limitation, not an inferred version. The current metadata-only
fix normalizes Torch's UUID to the `GPU-` selector expected by `nvidia-smi`
and explicitly records a failed query. It does not change numerical work,
inputs, outputs or timing samples, and was independently regression-tested.
No compiler flag or FMA-control guarantee is inferred from hardware branding.

For one paired A800 case (2 qubits, depth 2, noncommuting, seed 11), synchronized
complete-call medians are 0.037125 s for FP32, 1.088286 s for P4 and 0.013704 s
for FP64. Corresponding state maximum errors are approximately 1.43e-7,
2.51e-15 and 2.26e-16. These numbers characterize this eager small circuit;
they do not predict weak-FP64 hardware, compiled kernels or large circuits.
The reports and `frontier.json` retain all paired cases and measured peaks,
including cancellation cases where a strict tolerance is infeasible.

| Issue acceptance condition | Evidence and bounded conclusion |
| --- | --- |
| Complete matched circuits across precision, width and depth | Frozen manifest; 60 CPU and 60 CUDA results, both circuit families, matched parameters and coefficients; P2/P3 retained as controls. |
| Independent accuracy and supported gradients | NumPy C128 matrix/tangent oracle, finite differences and Decimal arithmetic checks; state/fidelity/norm/observable/gradient fields and separate synthesis/input/coefficient diagnostics. |
| Repeated absolute cost and frontier | Five synchronized samples per completed case, recomputed median, RSS/CUDA allocator peaks, payload separately scoped, per-workload accuracy/time/peak-memory frontiers. |
| CPU default and optional heterogeneous devices | CPU default, explicit CUDA/MPS invocations, real A800 and Apple M5 receipts; MPS stage refusals retained, actual output device/dtype guards, known and unknown host boundaries distinguished. |
| Preserve negative results and actual output identity | All 156 final records retained: 120 completed and 36 unsupported. Slower/inaccurate outcomes remain completed measurements; full last outputs and repeated word hashes distinguish bit identity from equal scalar errors. |
| Limits, tests and reproducibility | Binary32 exponent and supported gate/gradient limits above, source/manifest SHA-256 audit, pinned commands, regression tests and initial discovery archive. No capability promotion. |

All six conditions are met within this declared bounded development comparison.
Missing native internal tracing and unavailable driver-query data remain explicit
limits; they cannot support a device-only gate-synthesis or vendor certification
claim. Closing the characterization issue does not certify those capabilities.
