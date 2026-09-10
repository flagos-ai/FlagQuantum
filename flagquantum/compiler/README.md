# Compiler

This package transforms a Core-owned `CircuitIR` without executing it.
`pipeline.py` owns the stable optimization, layer scheduling, and topology-aware
compilation entry points. `routing.py` owns coupling maps and SWAP
routing. `openqasm.py` and `qcis.py` own their target-format emission.
`noise.py` owns the deterministic `CircuitIR + NoiseModel` to channel-bearing
`CircuitIR` transformation. `operator_lowering.py` owns the
internal backend/operator capability registry used before lowering or
serialization. `native_gate_legalization.py` validates evidenced native-gate
descriptors and applies the bounded, verified CircuitIR decompositions.
`topology_legalization.py` applies the existing router to one explicit coupling
map and verifies edge legality, restored output layout, bounded growth, and
deterministic evidence.
`schedule_legalization.py` constructs deterministic logical ASAP layers with
explicit wire and classical-data dependencies. Dynamic operations and channels
remain conservative barriers; this is not target timing or pulse scheduling.
`target_emission.py` gates the existing OpenQASM and QCIS text emitters behind
completed target legalization and binds deterministic emission audit facts. Its
result is not a new executable-artifact envelope.
`target_legalization.py` derives mandatory circuit requirements, checks one
explicit backend lowering, and matches one Core-owned target capability
snapshot without introducing a target IR or selecting a target.
`__init__.py` is the stable
expert-facing compiler interface.

`_hybrid/` owns the private structured-program semantic slice used to migrate
program-level control flow and ordered quantum effects into vNext. It is not a
second circuit compiler: its future quantum-region output is the existing
Core-owned `CircuitIR`, and it is intentionally absent from public exports.

Use `optimize(program)` for target-independent canonical optimization and
`compile(program, coupling_map=...)` for target-aware lowering. The pre-release name
`simple_compile` has been removed; it did not describe a distinct compilation
stage.

Run the user-facing optimization path from the repository root:

```bash
python -m examples.compiler_optimize
python -m examples.target_aware_compilation
```

Individual canonicalization functions are pipeline implementation details, not
expert-facing entry points. Change or compose them through `optimize`.

## Ten-minute change path

- Change local canonical optimization in `pipeline.py`.
- Change instruction layer scheduling in `pipeline.py`.
- Change coupling maps or SWAP routing in `routing.py`.
- Change noise-model lowering in `noise.py`.
- Change OpenQASM 2/3 target emission in `openqasm.py`.
- Change QCIS target emission in `qcis.py`.
- Change operator/backend lowering capabilities in `operator_lowering.py`.
- Change native gate matching and verified decompositions in
  `native_gate_legalization.py`.
- Change topology postconditions and routing audit in
  `topology_legalization.py`.
- Change dependency-preserving logical scheduling and its audit in
  `schedule_legalization.py`.
- Change verified static text emission and its identity binding in
  `target_emission.py`.
- Change capability-driven legality checks in `target_legalization.py`.
- Change private structured program semantics through `_hybrid/README.md` and
  its focused golden scenario; do not restore the historical `_compiler` tree.
- Run the compiler fixed-point, trainable-parameter, scheduler, routing, public
  namespace, noise, and CPU vertical-slice tests.

Runtime planning, backend selection, resource estimation, execution-plan
assembly, noise execution policy, provider lifecycle, and simulation numerics
do not belong here.
