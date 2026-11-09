# Public in-place modular inverse

`compiler/registers/modular/modular_inverse.py` ports the unchanged Qualtran v0.7.0
`KaliskiModInverse` compute/uncompute wrapper. Its body reuses
[inverse_algorithm](REGISTER_INVERSE_ALGORITHM.md) and the six existing
iteration layers. It emits native IR only, with no source-package or runtime
dependency. Copyright, Apache-2.0 attribution and the pinned source link remain
in the adapted module.

## Register ownership and actual outputs

`modular_inverse_ir(target, junk, modulus, workspace, classical_bits,
adjoint=False)` takes an n-bit big-endian target, 4n logical history bits,
5n+8 clean work wires and max(2n-1,n+1) reusable classical slots. Work contains
u/r/s (n bits each), active f (one bit) and the shared 2n+7-wire body pool.
Forward junk and work start zero. Valid target inputs are zero or units modulo
positive odd p, with n>=2 and p+1<2**n.

Output is `x^-1 * 2^(2n) mod p`; zero maps to zero. The first 2n junk bits retain
the original marker history, and the last 2n retain the terminal history. All
workspace is zero on the actual returned work roles. Classical slots hold
recent outcomes, not a promised zero value.

The source's logical in-place signature does not mean a fixed physical output.
`modular_inverse_outputs(target, workspace)` resolves the real result wires:
forward result comes from the body's rotated coefficient word, and the original
input target becomes the returned clean auxiliary r word. No SWAP is inserted
just to restore original names. The inverse consumes those exact output/work
roles with the matching junk; it restores original target wires and clears both
junk and workspace. Output mapping does not depend on the modulus.

The history is essential logical data. For n=3,p=5,x=1, the source returns value
4, marker history 36 and terminal history 4. Uncomputing value 4 with zero junk
instead of `(36 << 6) | 4` fails to restore the input and leaves nonzero
history/work. The original raw gate behavior is preserved and tested.

## Actual source graph and numerical evidence

The public source uses `bb.add_from`, inlining the entire body into its graph.
The oracle walks that actual graph rather than replacing it by an assumed body
call. Root allocation bindings are traced backward from the first original
iteration's input edges through same-port initialization nodes to the actual
Allocate nodes. Every Allocate is accounted for; independent allocation order
never assigns semantic roles. Every original Free is matched against its exact
expected logical buffer. Allocate and Free emit no gates; zero promises are
proved by the full channel tests instead of being hidden by reset.

The body source leaf reader is extracted into
`tests/unit/inverse_algorithm_source.py`, shared by the body and public wrapper
oracles. It retains original component and per-wire gate/feedback assertions.
This is test infrastructure, not a claim of a new production graph API.

Forward and uncompute complete original gate/operand/feedback/physical-port
streams agree at n=2,p=1; n=3,p=5; n=4,p=9; n=8,p=251. Every valid unit and zero
at those moduli, n=3,p=3 and n=4,p=13 has the exact coefficient, original logical
history, clean work and inverse restoration for every erasure history under the
existing literal-primitive composition certificate. After uncompute, all 4n
junk bits and every clean work bit are zero.

Primitive numerical tests use the existing G1 sparse test executor, with all
valid inputs coherently entangled with a spectator, at n=3,p=5; n=4,p=9; and
n=8,p=251. Three seeded histories per native direction plus one original-source
replacement per direction compare every complex amplitude, including unexpected
support/cancellation residuals, at absolute 2e-12 and relative zero. Each measured
conditional probability is one half. They validate complete occupied vectors
without allocating 39/49/89-qubit dense states. This is not exhaustive numerical
history enumeration or a production sparse backend. The all-history certificate
retains the eight-bit exact branch probability as 2^-1111.

## Measured streams and independent models

The public wrapper adds no physical gates to the full body. Both directions
have identical primitive counts, with T including T-dagger. Every measured erase
also contributes one S and one conditioned CZ.

| n, p | Measured T | Measured CX | Measured X | Measured H | Measurements / resets | Clean work | Logical junk | Total qubits | Classical slots | Default model T | Exact-swap model T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2, 1 | 592 | 1,086 | 257 | 327 | 85 | 18 | 8 | 28 | 3 | 484 | 592 |
| 3, 5 | 1,292 | 2,383 | 458 | 696 | 176 | 23 | 12 | 38 | 5 | 1,040 | 1,292 |
| 4, 9 | 2,260 | 4,157 | 719 | 1,201 | 299 | 28 | 16 | 48 | 7 | 1,804 | 2,260 |
| 8, 251 | 8,812 | 16,308 | 2,181 | 4,581 | 1,111 | 48 | 32 | 88 | 15 | 6,940 | 8,812 |

The default source model weights CSwap at four T, while the unchanged executable
swap decomposition uses seven T. The exact-swap model agrees with measured
primitive T. Independent source qubit and measurement models also agree;
estimates and actual streams remain distinct.

## Local acceptance

The wrapper adds 25 tests. All 701 concentrated dependency regressions pass
without skips. After extracting the shared source reader and removing an
unreachable source-adapter branch, a final 56-case body/wrapper regression passes;
the wrapper and both source readers have 100% statement coverage. Strict types
pass on 971 production files, the example and both readers. Minimum/current
Black, Ruff, compiler scope, architecture and multi-level IR checks pass. The
source Git blob remains unchanged. These are scoped checks, not a rerun of every
historical G2 case.

## Runnable native append example

```bash
python -m examples.algorithms.register_modular_inverse
```

The user writes `circuit.append(ModularInverse())` and then its matching inverse.
The example prints actual physical result/work roles, retained history and
primitive counts, and separates default/exact model estimates. It constructs
38-wire IR without a dense allocation and requires neither Qualtran nor NumPy.
Complete numerical execution evidence is the primitive test path above.

Source limitations remain explicit: all-ones modulus p=2**n-1 fails the original
AddK(p+1) load; nonunits cannot promise logical cleanup; a one-bit termination
layer fails the original minimum-control requirement. Symbolic graph/resource
interfaces, the upstream examples/notebook/test-file inventory and other G2
families remain open. This concrete compute/uncompute acceptance does not mark
the whole source file or G2 complete.
