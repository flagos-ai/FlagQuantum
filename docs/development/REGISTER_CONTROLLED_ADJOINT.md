# Exact controlled arithmetic adjoints

Run `python -m examples.algorithms.register_controlled_modular_adjoint` without
Qualtran or NumPy. `circuit.append(ControlledModularAddition(inverse, cv))` keeps
the original control/x/y ports, exercises both control polarities on coherent
complex inputs (including x=p), checks clean work and restores inputs in both
composition orders through the existing dynamic runtime.

## Ownership and shared controlled-add body

`controlled_add_adjoint_ir` has the same signature/roles as controlled_add_ir:
control, left, right, loaded, carries, load_bits, carry_bits, and control_value.
For m left bits and n>=m right bits, supply m loaded wires/slots and n-1 clean
carry wires/slots. Both directions share validation, control-polarity And load,
reverse-order erasure and normalization policy. Only the middle Add changes to
the original exact Add adjoint. Forward streams remain checked by existing
source regressions. No compiler runtime, device choice, root public API or new
representation is introduced.

Preserve both inputs/control, update right to right-left modulo 2**n on the
selected branch. All quantum work starts/ends zero. Equal-width signed bit
patterns are supported; unequal-width patterns use zero extension, retaining
the source executed circuit. Do not interpret that as a fix of the source's
unequal-width signed classical-model mismatch (ledger Q-0186.02).

The exact adjoint is not complemented-forward resynthesis: at three bits the
literal positive-controlled adjoint has no X gates, while complement/CAdd/
complement adds six Xs. Gate-level equality to the original source is tested,
beyond equality of the mathematical subtraction.

## Original controlled modular-add adjoint

`controlled_modular_add_adjoint_ir(control,left,right,modulus,workspace,
classical_bits,control_value=...)` belongs to controlled_modular_addition_adjoint.
It consumes the existing _controlled_modular_workspace layout and keeps source
ports. The promised inputs in both branches are x in [0,p], y in [0,p),
0<p<2**n. It implements y=(y-x)%p on the selected coherent control branch.
Work/slots remain max(2n+2,n+6) / max(2n,n+2).

Preserve source reverse components:
X(sign), CLinearDepthGreaterThan adjoint, sign-controlled AddK(p) adjoint,
extended AddK(-p) adjoint, CAdd adjoint. Original constants are retained; no
measured primitive stream is reversed. The comparator's decomposition is a
palindrome of sign extension, complement, OutOfPlaceAdder/adjoint and a
self-adjoint MultiControlX. Tests independently compare its actual source
adjoint gates before reusing the existing controlled_greater_ir. Its inverse
source allocation discovers the b-prefix before a-prefix: the reference binds
these temporary names explicitly, without changing source logic or allocating
another native pool.

This is distinct from original CModSub's complement/load/CModAdd/inverse-load
construction. On their common promised domain both have the same ideal logical
map, but different primitive streams and measurement histories. Existing
controlled_modular_subtract_ir is retained unchanged; its counts are not
relabeled as the exact CModAdd-adjoint counts.

## Pinned source and complex channels

Unmodified Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389,
controlled_addition.py::CAdd and mod_addition.py::CModAdd and their original
Adjoints are the references. Google Apache-2.0 attribution is retained.
Tests compare gate multisets, per-wire ordered streams, operands and feedback
for both control polarities and concrete source dtypes. Independent gates may
commute. Reference adapters retain component lifetimes and expand original
Adjoints, including controlled AddK adjoints, before generic Cirq synthesis.

Every tested valid basis input is checked by the strict literal-And channel
certificate; positive unit-amplitude compute columns and both erase outcomes
were independently verified. By linearity/composition all outcome histories
have the ideal full complex map times 2**(-measurements/2) and zero work.
CAdd exhaustive inputs cover equal/unequal widths through eight, with all raw
measurement outcomes at widths 1/1 and 2/3 on entangled spectator inputs.
CModAdd inverse covers all 126,504 valid n=8,p=251 control/x/y inputs, including
x=p. Raw projection enumerates all 64 one-bit branches and selected
zero/one/alternating two-bit branches. Higher-width coverage is a composition
proof, not exhaustive dense trajectory execution. Forward-after-adjoint
restoration and the runnable runtime example independently exercise composition.

## Measured resources and separate source estimates

T includes T-dagger. Conditional CZ counts below are emitted instructions;
runtime execution depends on outcomes. One reset and S per measurement, H=3m.

| n=8 circuit | cv | CX | X | T | Measurements | Clean work | Total qubits | Classical slots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CAdd adjoint (equal widths) | 1 | 129 | 0 | 60 | 15 | 15 | 32 | 15 |
| CAdd adjoint (equal widths) | 0 | 129 | 32 | 60 | 15 | 15 | 32 | 15 |
| CModAdd adjoint, p=251 | 1 | 479 | 45 | 164 | 41 | 18 | 35 | 16 |
| CModAdd adjoint, p=251 | 0 | 479 | 81 | 164 | 41 | 18 | 35 | 16 |

Independently evaluated source QECGatesCost gives CAdd-adjoint 4*(m+n-1) T and
m+n-1 measurements; CModAdd-adjoint gives 20n+4 T and 5n+1 measurements.
QubitCount agrees on tested layouts. Primitive Clifford inventory remains
separate from abstract source And/Clifford model fields. At positive control
and n=8, the source reports And/Clifford=15/54 for CAdd-adjoint and 41/320
for CModAdd-adjoint; neither Clifford field is relabeled as a primitive total.

For comparison, the already ported original CModSub at n=8,p=251 emits
220 T / 55 measurements, because it has a different original call chain.
A dedicated three-bit comparison verifies equal logical outputs but distinct
streams and measurement counts (64 T/16 versus 80 T/20). It does not assert
identical instruments with different-length classical histories.

Inherited source bounds, unequal-width signed-model inconsistency, symbolic and
generic multi-control paths remain explicitly outside this concrete acceptance.
