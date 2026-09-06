# Compiler

This package transforms a Core-owned `CircuitIR` without executing it.
`pipeline.py` owns the stable optimization, layer scheduling, and topology-aware
compilation entry points. `routing.py` owns coupling maps and SWAP
routing. `noise.py` owns the deterministic `CircuitIR + NoiseModel` to
channel-bearing `CircuitIR` transformation. `__init__.py` is the stable
expert-facing compiler interface.

Use `optimize(program)` for target-independent canonical optimization and
`compile(program, coupling_map=...)` for target-aware lowering. The pre-release name
`simple_compile` has been removed; it did not describe a distinct compilation
stage.

Individual canonicalization functions are pipeline implementation details, not
expert-facing entry points. Change or compose them through `optimize`.

## Ten-minute change path

- Change local canonical optimization in `pipeline.py`.
- Change instruction layer scheduling in `pipeline.py`.
- Change coupling maps or SWAP routing in `routing.py`.
- Change noise-model lowering in `noise.py`.
- Run the compiler fixed-point, trainable-parameter, scheduler, routing, public
  namespace, noise, and CPU vertical-slice tests.

Runtime planning, backend selection, resource estimation, execution-plan
assembly, noise execution policy, provider lifecycle, and simulation numerics
do not belong here. The private `_compiler` research tree remains frozen until
one of its implementations replaces a stable compiler concern under the same
consumer-facing tests.
