# Compiler convergence inventory

Status: complete

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

The former `flagquantum.compilation` transition package has also been removed.
Execution-plan products, validation, serialization, and calibration now live with
their Runtime owner. No second compiler authority remains in the package tree.

## Ten-minute path

- Public compiler surface: `flagquantum/compiler/__init__.py`
- Optimization and scheduling: `flagquantum/compiler/pipeline.py`
- Topology routing: `flagquantum/compiler/routing.py`
- Noise lowering: `flagquantum/compiler/noise.py`
- Runtime orchestration: `flagquantum/runtime/planner/`
- Execution-plan product: `flagquantum/runtime/execution_plan.py`

The CPU vertical slice must continue to pass without importing a private compiler
tree or any vendor SDK.
