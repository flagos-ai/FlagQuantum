# Controlled addition with a mixed carry input

G2-KADD07 owns the locked `gen_iadd_control` and `gen_isub_control` schedules
in Compiler registers. The interface takes a big-endian quantum target and
quantum offset, a quantum enable, optional constant/classical/quantum carry,
caller-owned clean carries and a separate reusable classical result slot.
The enabled result is target + offset + carry modulo the target width. The
subtraction entry uses the source NOT conjugation with carry false. Wider
offset tails above the target width remain unchanged; shorter offsets refuse,
matching the locked source. All quantum inputs and clean wires are disjoint.

For width n >= 2, supply n zero clean wires. They return to zero; quantum
carry, enable, offset and any unused clean wires are restored. Carry bits
are read-only and must be defined by the caller. The measurement slot must
not alias a classical carry input. Source reset operations of clean carries
are represented by the explicit clean-input precondition. There is no new
Core opcode, public export, or separate execution implementation.

Reference: Kickmix commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`,
`gen/adders/gen_iadd_control.cc`, blob
`ee8062ae063ce241383bdb633ead74c4dafc5db1`. Preserve emitted loop operand
order, final carry cleanup and all phase feedback, not just the arithmetic map.

## Stream fidelity and measurement evidence

The source LoopBuilder uses an exclusive loop bound. Its carry loops run over
indices 1 through n-2; the last of the n requested clean wires is unused.
Loop-emitted CCX controls follow the native operand order (offset before target
for carry construction, offset before enable for updating sums). The leading
and trailing non-loop calls preserve their original control order.

The source fixture contains 180 complete primitive stream digests and category
counts: widths 0–8, equal/two-extra offset widths, minimal/two-extra workspace,
all four carry kinds and carry-false subtraction. Digests omit register/INFO
annotations and reset of supplied zero carry inputs. Every primitive, ordered
operand and effective classical guard set is retained. Read-only positive
condition scopes are flattened to their effective guard sets; this does not
reorder gates or remove repeated gates. The result bit is separate from the
caller classical carry, including the conditional NEG in its final erasure.

Regenerate the fixture from the unchanged pinned source:

```sh
python tests/unit/fixtures/generate_kickmix_controlled_addition.py \
  --source-root /path/to/locked/kickmix \
  --output /tmp/kickmix_controlled_addition.json
```

The shared test-only C++ oracle builder validates the generator blob and compiles
native gate implementations, including the pure C++ definitions under
`py/circuit/ops`. The increment fixture generator also uses this builder; its
regenerated JSON is byte-identical. No upstream implementation is rewritten.

All small measured branches on arbitrary complex superpositions of targets,
offsets, enable and quantum carry are checked against the independent modular
arithmetic permutation. Offset tails, constant/classical/quantum carry, inactive
enable, subtraction and unused work are covered. Native basis tests preserve
absolute phase and verify every clean wire is zero. Classical carry inputs are
unchanged. These tests do not rely on sampling every branch.

For n >= 2 the emitted stream has 2n-1 CCX and n-1 HMR, with n-1 measured CZ
corrections. Other Clifford/scalar phase costs depend on the carry kind and
subtract conjugation. The three-bit quantum-carry example emits 5 CCX, 12 CX,
2 HMR, 2 CZ and 2 Z. It requires three supplied clean wires, uses two of them,
and reuses one classical measurement slot. These are measured native primitive
counts; no synthesized T count or analytical model value is substituted.

Run `python -m examples.algorithms.controlled_carry_addition` for the
`circuit.append(op)` example. There are 68 new tests; the concentrated
addition, predicate, classical-lowering and native-format regression passes
784 tests without skips. Strict mypy covers the emitter, example and source
fixture tools. Black, Ruff, compiler scope and architecture checks pass.
