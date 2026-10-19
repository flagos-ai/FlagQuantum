# Static / ideal-circuit QCIS import

This experimental module owns conversion of static numeric or parameterized QCIS text into
FlagQuantum CircuitIR. It depends only on Python's standard library, Core IR,
and the existing Ecosystem conversion contracts. It must not invoke a compiler,
container, runtime, simulator, or remote service. No producer-specific behavior
belongs here.

## Ten-minute path

```python
import flagquantum as fq
from flagquantum.ecosystem.qcis import import_qcis

converted = import_qcis("X Q17\nM Q17")
result = fq.run(converted.ir)
assert result.probabilities.tolist() == [[0.0, 1.0]]
```

Change a mapping in `conversion.py`, add an independent input/result scenario,
then run:

```sh
python -m pytest tests/unit/test_qcis_import.py tests/integration/test_qcis_import.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

## Supported semantics

`import_qcis(source: str, *, allow_lossy: bool = False)` returns the existing
`InteropImportResult`. This domain entry point adds no Stable Core export or
adapter registry entry and introduces no new public result or exception type.

This is not complete QCIS support. `report.lossless` describes conversion of
the supported **ideal circuit semantics** only. It does not certify preservation
of full QCIS hardware semantics, calibration, scheduling, pulse behavior, or
device execution. Even lossless gate import is not a hardware execution claim.

- Native gates: X2P, X2M, Y2P, Y2M, RZ, CZ.
- Composite gates: X, Y, Z, H, S, SD, T, TD, RX, RY, RXY.
- Angles: finite decimal/scientific-notation radians, including signed values.
  Symbolic angles use the restricted expression grammar below.
  Half rotations are RX/RY(+/-pi/2), not phase-shifted square-root gates.
- RXY Qn phi theta rotates about (cos(phi), sin(phi), 0). It lowers to
  RZ(-phi), RX(theta), RZ(phi) in execution order, preserving global and
  relative phases without adding an IR opcode.
- Terminal M with one or more distinct qubits, across one or more lines. Output
  qubits follow measurement appearance order, independently of label order.
  QCIS does not specify shots, so the IR requests exact Z-basis probabilities.
  For finite sampling use `fq.Circuit.from_ir(converted.ir)` with
  `outputs=fq.counts(converted.ir.measurements[0].wires)` and explicit `shots`.
  An existing IR measurement is never overwritten by runtime output arguments.
- Blank lines, CRLF, whitespace, and `#`/`//` comments. Mnemonics and Q labels
  are uppercase; labels have no leading zeroes except Q0 itself.
- Q labels are sorted numerically and compacted. `metadata['interop']['qubits']`
  lists the original label at each IR index. Unmentioned/idle qubit extents
  cannot be recovered from QCIS and are not invented.
- I Qn t and B require explicit `allow_lossy=True`: idle becomes identity and
  barrier alignment is omitted. Original text, line, and integer duration ticks
  (0.5 ns per tick) remain as annotations, and the report is not lossless.
  This is ideal gate simulation, not hardware timing or noise simulation.

Empty input, invalid arity/labels, duplicate operands/measurements, nonterminal
measurement, unknown gates, reset, control flow, pulse instructions, and
expressions outside the supported grammar fail with `InteropConversionError`.
`allow_lossy=True` never authorizes dropping unsupported instructions.
Diagnostics include physical source line numbers and operation names.

Definitions: <https://docs.quantumcomputer.ac.cn/Appendix/C/> and
<https://docs.quantumcomputer.ac.cn/FAQ/4/>. Composite gates follow the defined
ideal matrices; hardware decompositions may differ by an overall global phase.
The imported IR is an input program, not a compiled executable artifact.

Independent semantic tests use handwritten QCIS and analytic statevector oracles.
Running importer tests never invokes a producer, container, or download.

## Parameter expressions and binding

The importer supports this explicit expression subset for RX, RY, RZ and both
RXY angles. It is not a claim that every QCIS producer accepts these expressions.

- Finite decimal/scientific numbers, `pi`, and case-sensitive parameter names
  matching `[A-Za-z][A-Za-z0-9_]*`. `pi` is reserved; non-finite spellings such as
  `nan`, `inf` and `infinity` are rejected regardless of case.
  Python keywords such as `lambda` and `class` are valid parameter names,
  including in expressions like `lambda/2`. Python literals `True`, `False`
  and `None` remain unsupported.
- Parentheses, unary `+`/`-`, and binary `+`, `-`, `*`, with ordinary precedence.
  Division is supported only by a nonzero constant expression, such as
  `theta/2` or `theta/(2*pi)`. Symbolic division, powers, calls, attributes,
  indexing and executable code are rejected. `//` starts a QCIS comment.
- Operands are separated by whitespace. Enclose an angle containing spaces in
  parentheses: `RXY Q7 (phi + pi/2) (2 * theta)`. Expressions without spaces
  need no outer parentheses. Expressions are bounded to 4096 characters,
  256 AST nodes and 32 recursive AST levels.

Names become existing Core `Parameter` objects; operations become existing
`ParameterExpression` values. Constant subexpressions are folded and checked
for finiteness. Repeated names share one binding, including across decomposed
RXY gates. IR JSON serialization preserves these expressions. No `eval`, new
parameter registry, IR schema change or producer dependency is involved.

```python
import torch
import flagquantum as fq
from flagquantum.ecosystem.qcis import import_qcis

converted = import_qcis("RXY Q7 phi (theta/2)\nM Q7")
circuit = fq.Circuit.from_ir(converted.ir)
theta = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
bound = circuit.bind_parameters({"phi": 0.2, "theta": theta})
result = fq.run(bound, outputs=fq.probabilities((0,)))
result.probabilities[0, 1].backward()
assert theta.grad is not None
```

`Circuit.from_ir` copies gates, not terminal measurements. When using this
binding workflow, request the source measurement wires explicitly with
`fq.probabilities(...)`, `fq.samples(...)` or `fq.counts(...)`; do not assume
conversion to Circuit retains `M`. Numeric IR can still be passed directly to
`fq.run` with its terminal measurements intact. Missing bindings fail through
the existing parameter API. Bindings and their device/dtype/gradient behavior
belong to Circuit and the selected executor; the importer does not execute them.

CPU execution and first-order PyTorch gradients are tested against independent
analytic values, including shared nonlinear parameter expressions. This is not
a claim of GPU, distributed training or pulse execution support.
