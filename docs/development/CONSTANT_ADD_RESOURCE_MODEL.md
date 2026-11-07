# Constant-addition native resource chain

The register compiler owns the AddK resource chain in
`arithmetic_declarations.py` and its concrete/symbolic consumers in
`constant_add_resource_model.py`. Existing `arithmetic.py` and `always.py` own
emission and inherited-control bypass. This batch does not change those gates.
No SDK objects enter the native compiler contracts.

The reference is unmodified Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`:

- `arithmetic/addition.py`: AddK's load/add/unload and Counter declaration;
- `arithmetic/bitwise.py`: XorK's dtype encoding and symbolic dispatch;
- `bookkeeping/always.py`: Always's direct delegation to the child's graph.

The original direct Counter is `XorK: 2, Always(Add): 1`, in that order. The two
XorK calls are self-adjoint and share one key. Always(Add) retains its native
identity and directly delegates Add's child declarations. There is no invented
Always-to-Add edge. Adjoint wrappers preserve the declared row order and invert
the Add children. The existing emitter's Always wrapper still bypasses inherited
controls; resource metadata does not execute or implement that protocol.

## Interfaces and ownership

`source_constant_add_resource_formula(width, constant, encoding=..., adjoint=...,
legacy_shims=...)` consumes the native graph and shared Add model. The encoding
is unsigned, two's complement, or default Montgomery with no modulus parameter.
`source_constant_add_cost` returns the same declaration as a concrete ComponentCost
for a Python integer width. A symbolic constant may still produce numeric shape
counts; that is not a claim that its gate sequence can be emitted.

Arithmetic keys retain the original width and constant objects. Python negative
constants are normalized modulo 2**n only for unsigned/default Montgomery load
keys; signed constants remain signed. Concrete XorK uses the legal dtype bit
pattern's population count. If either width or constant is symbolic, the original
XorK declaration counts n X gates. Concrete range failures remain failures and
are not silently wrapped. Sympy Integer objects retain the symbolic branch.

The existing Add model supplies all seven categories and its peak. The graph
query uses the established static-child-cost boundary to consume that model at
Always(Add), without repeating Add equations. Tests independently replace the
Add model and the graph recipe without modifying the consumer.

The target has n bits. Constant loading owns n clean work bits and Add owns n-1
carry bits in concrete execution: peak 3n-1. Symbolic width retains the original
signature/callee maximum, including Max(3, n, 2n) for unconstrained n. Both Python
and Sympy zero-width AddK have failing original QubitCount queries and are refused
by the combined estimate. Their direct declarations remain inspectable.

## Distinct source paths and count evidence

These default-profile values match the locked source. A symbolic constant is not
silently converted to a Python integer even when its printed value is numeric:

| Width object | Constant object | Model And | Model Clifford | Model measurements | Peak | Emitted T/Tdg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Python 3 | Python 5 | 2 | 15 | 2 | 8 | 8 |
| Python 3 | Sympy Integer(5) | 2 | 17 | 2 | 8 | not emitted |
| Sympy Integer(3) | Python 5 | 2 | 17 | 2 | 6 | not emitted |
| positive symbol n | symbol k | n-1 | 9n-10 | n-1 | Max(3,2n) | not emitted |
| Python 1 | Python 0 | 0 | -3 | 0 | 2 | 0 |

For three bits and Python 5, the primitive stream has 4 X, 21 CX, 6 H, 2 S,
8 T/Tdg, 2 measurements, 2 conditional CZ and 2 resets. Primitive counts match
the original decomposed stream, while model Clifford counts remain abstract.
The one-bit source model declares -3 Clifford although the actual sequence has
one CX; that original discrepancy is retained. No zero-constant optimization
removes the original Always(Add) sequence.

Upstream objects can compare Python and Sympy numeric constants equal despite
these different resource results. Native arithmetic keys distinguish their
source-dispatch kinds to prevent cache collisions. Reference comparisons use
separate upstream queries without changing its equality or source code.

The tested gate streams cover unsigned, signed and default Montgomery inputs,
both directions, and full operand/feedback dependencies. Small coherent inputs
include an entangled spectator. Every measurement history is projected without
normalization and compared with source classical permutations, including branch
probabilities, relative phases and all constant/carry work returning to zero.
Symbolic models do not imply an executable symbolic circuit. Controlled AddK's
existing gate path remains independently tested; its symbolic raw resource graph
is outside this slice.

Run:

```sh
python -m examples.algorithms.arithmetic_resource_graph
python -m examples.algorithms.register_constant_addition
python -m pytest -q tests/unit/test_constant_add_resource_model.py tests/unit/test_register_constant_addition.py tests/unit/test_constant_addition_adjoint.py
```

The executable example continues to use `circuit.append(op)` and now reports the
shared logical model separately from emitted primitive counts.

This batch adds 327 tests and passes 1,408 concentrated checks without skips.
The three source blobs match the locked inventory. Changed production modules
and both runnable examples pass strict mypy, Black, Ruff and architecture checks.

## Controlled source identities (G2-A47)

The single-control native graph follows the source's actual wrapper classes:
control zero delegates to positive controlled AddK once and declares two X gates
at that outer level. Positive controlled AddK declares two positive controlled
XorK loads and one Always(Add). Positive controlled XorK declares CNOT with the
same concrete population-count or symbolic-width multiplicity. No And predicate
is introduced for one control, and open-control X costs are not multiplied per
loaded bit. Adjoint declarations retain the source wrapper and row order.

`source_controlled_constant_add_resource_formula` and
`source_controlled_constant_add_cost` reuse the same model consumer with an
explicit control polarity. All seven Add categories pass through the existing
static-cost interface. With Python width and constant, the peak includes one
live control alongside Add. If either object is symbolic, the original fallback
is Max(n+1, 2, Add peak), without adding another fictitious lifetime.

For width Python 3 and constant Python 5 the controlled peak is 9; replacing
only the constant with Sympy Integer(5) gives peak 8; replacing the width with
Sympy Integer(3) gives peak 6. The default-profile Clifford estimates for control
one are respectively 15, 17, 17; control zero adds exactly two. The original
one-bit diagnostics remain distinct from emitted primitive counts. Both
zero-width combined queries still fail explicitly. These declarations preserve
source behavior rather than normalizing symbolic literals into concrete inputs.

Controlled graph tests compare every raw row, both profiles/directions, three
encodings, and all small coherent control/target/spectator measurement histories.
The existing controlled emitter is unchanged. Multi-control graph support and
arbitrary Always resource children remain outside this implemented slice.

G2-A47 adds 610 controlled-resource tests; the concentrated batch passes 2,018
checks without skips. The restored persistent reference runner verifies source
blob IDs and the locked Qualtran/Kickmix versions before launching pytest.
The isolated restored dependencies affect reference tests only; project
requirements and numerical execution code are unchanged.
