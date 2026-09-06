# Compiler convergence inventory

Status: stable Compiler authority established; transitional execution-plan package remains

Updated: 2026-09-07

## Current authority

`flagquantum.compiler` is the only compiler implementation in the product tree. It
owns target-independent optimization, instruction scheduling, topology routing,
backend lowering, and noise-model lowering. Runtime calls this package through
`compile`, `optimize`, and the focused routing/noise entry points.

The disconnected `flagquantum._compiler` research candidate and its phase-specific
tests, fixtures, and benchmarks were removed before release. Its useful ideas remain
available in Git history, but unsupported private contracts and deployment/runtime
shadows no longer appear as a second implementation authority.

## Remaining boundary

`flagquantum.compilation` is not a second compiler. It temporarily contains the
protected `ExecutionPlan` product, execution-plan serialization and validation,
performance calibration, and noisy-plan data models used by Runtime. Program
transformation modules have already moved to `flagquantum.compiler`; planning and
selection implementations have moved to `flagquantum.runtime.planner`.

The remaining convergence step is narrow:

1. move execution-plan ownership to its final Core/Runtime boundary without changing
   the public execution contract;
2. update Runtime imports;
3. delete `flagquantum.compilation` once no production caller remains.

No new compiler IR, pass framework, capability vocabulary, artifact envelope, or
compatibility layer should be introduced for this move.

## Ten-minute path

- Public compiler surface: `flagquantum/compiler/__init__.py`
- Optimization and scheduling: `flagquantum/compiler/pipeline.py`
- Topology routing: `flagquantum/compiler/routing.py`
- Noise lowering: `flagquantum/compiler/noise.py`
- Runtime orchestration: `flagquantum/runtime/planner/`
- Transitional plan product: `flagquantum/compilation/`

The CPU vertical slice must continue to pass without importing a private compiler
tree or any vendor SDK.
