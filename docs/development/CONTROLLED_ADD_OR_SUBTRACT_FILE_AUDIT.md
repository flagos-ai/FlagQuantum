# Conditional addition/subtraction whole-file audit

The complete `arithmetic/controlled_add_or_subtract.py` and its original test file
are audited at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Both original Git blobs are asserted.
The source defines one class, its default b dtype, signature, composite body and
adjoint, three factories and one documentation specification.

## Interface and direct dependency closure

Compiler `registers/arithmetic/addition.py` owns
`measurement_add_or_subtract_ir`. The authoritative Add owner supplies its
middle operation. Shared controlled constant XOR supplies the two controlled
BitwiseNot shells, with the all-ones target pattern. X supplies the optional
outer control flips. Existing bitwise and addition owners remain replaceable by
the same consumers. The separate unitary ripple path keeps its own contract.

Physical binding is MSB-first ctrl/a/b with explicit carry qubits and classical
slots. All logical ports are through. Carries start and finish in zero, and each
has a distinct feedback slot. Original b dtype defaults to a dtype. A true
`add_when_ctrl_is_on` adds on control one. False reverses those sectors. The
source argument prose describes False inconsistently, so the actual body and
adjoint define the preserved behavior. The adjoint flips the polarity while
retaining an ordinary forward Add in the middle.

| Complete original definition | Native disposition and evidence |
| --- | --- |
| Fields, default, signature | Existing native physical binding, typed register owners and actual source capture retain original ctrl/a/b order and width. Both complete unsigned/signed eight-bit factories and their DocSpec membership are checked. |
| Composite body | Both complete factories are captured independently in each original direction. Every gate, operand and measurement feedback is compared with the original reference traversal. Existing direct native lowering tests cover eight equal/unequal width pairs, all three integer dtypes and both polarities. |
| Adjoint | Actual captured roots retain opposite polarity, original Add and exact source stream. Small coherent tests cover both initial polarities, both directions, all four first histories and all sixteen compute/inverse history pairs, retaining unnormalized branch amplitude and zero carries. |
| Concrete factories | Both eight-bit factories pass named capture, workspace/slot ownership, exact stream, independent model and CircuitIR JSON through ordinary append. Full eight-bit dense state enumeration is not claimed. Existing eight-bit boundary witnesses and branch certificates are reused. |
| Symbolic factory | Original signed n-bit metadata and model are retained. The source decomposes one level to five subbloqs. Physical capture rejects the symbolic port width. This is separate from symbolic resource estimation. |
| Documentation specification | All three factories keep their original source identity. Existing native example `examples.algorithms.register_controlled_add_or_subtract` demonstrates coherent append, measured execution, exact inverse and zero work without a new API. |

The new captured small tests use unsigned, signed, Montgomery and mixed
unsigned/signed words, reordered physical ports and a coherent spectator. They
compare complete complex output states against the original tensor, all raw
measurement histories, clean work and inverse restoration. The wider signed
input remains subject to the source Add contract described below.

## Complete original test disposition

The original file has six test functions and one naive reference class. The
class is intentionally a Bloq with an attrs constructor and is not itself a
pytest test class. `test_examples` covers all three factories. Classical
simulation checks all unsigned words at widths two, four and five.
`test_caddsub_mixed_dtype` covers Add and conditional Add at an occupied signed
result. `test_tensor` compares the complete source circuit with the naive
controlled Add/Subtract/Negate construction at widths three and four.
`test_t_complexity` checks the symbolic n-1 CCZ estimate.
`test_controlled_t_complexity` evaluates both orders of outer control and
adjoint at width ten. These original assertions run without reference edits.
Native exact stream, dtype, coherent branch and resource consumers provide the
separate execution evidence.

## Resource accounts and remaining boundaries

Existing eight-bit native execution reports 28 T/TDG, 97 CX, zero or two X,
seven measurements and seven clean carry qubits, with 24 total physical wires
and seven slots. Independent original models report 28 T-equivalent, seven
measurements and peak 24. The one-bit original Clifford model undercounts its
executed stream by four gates. That diagnostic remains recorded and is not
corrected in the reference. The existing independent formula is compared over concrete and symbolic equal/unequal widths,
all category fields and peak width, separately from the measured inventory.

Q-0183.03 remains open. Original QInt(1)/QInt(2) with control one, a=-1 and b=0
has a classical result -1, whereas its gate circuit and native lowering return
01. The source Add zero-extends that narrow signed input. This batch preserves
the actual quantum behavior and does not claim mathematical signed extension.
Generic symbolic typed Program admission and outer controlled/static foreign
conversion retain their existing boundaries. Q-0183 and Q-0184 remain open for
these source/admission issues and the original unimplemented test mode.
Logs are `ftqc-reference-env/g2-controlled-add-or-subtract-file-*.log`.

The concentrated batch passed 222 cases without warnings or skips. Strict typing
of the new source-capture test, formatting, lint, Compiler scope, architecture,
multilevel IR and protected API checks passed. The untouched original file
reports 12 passed and one skipped count-comparison fixture mode. Its two warnings
retain the original unknown slow marker and the attrs reference class that
pytest cannot collect. The original test class remains a reference Bloq.

The arithmetic and conversion export responsibility tables were corrected in
this batch to point SignedIntegerToTwosComplement's declared one's-complement
input to `ones_complement_to_twos_ir`. They had incorrectly named the separate
sign-magnitude path. The existing signed-encoding code and tests already separate
those contracts and remain unchanged by this documentation correction.
