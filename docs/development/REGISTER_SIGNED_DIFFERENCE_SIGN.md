# Pinned signed difference-sign path

`compiler/registers/linear_comparison.py` now shares the original concrete
`LinearDepthGreaterThan(signed=True)` carry program with the unsigned path.
The original is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
[`comparison.py`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py).
The existing Google copyright and Apache-2.0 attribution are retained.
Only physical register binding and primitive IR emission change; the carry
program, original adjoint, operands and measurement feedback are retained.

## Ownership and contract

The compiler emits IR. It owns no simulator, device selection or runtime.
Both inputs are equally sized, big-endian words. The target is an arbitrary
XOR target. The caller supplies clean carry wires and distinct classical slots;
all input wires and quantum work are restored. For width `n>=2`, signed mode
uses `n-1` work wires and slots, without the unsigned path's two sign prefixes.
The unchanged default remains a full unsigned comparison.

**Signed mode is the original difference-sign program, with a restricted
mathematical comparison domain.** It XORs the most significant bit of
`(b-a) mod 2**n`. This equals two's-complement `a>b` only without subtraction
overflow. At width one the original special case ignores signed mode and
computes unsigned `a>b`. Do not replace a full signed comparator with this path.
The existing `controlled_comparison.py` sign-extends before subtraction and is
a separately validated consumer; its implementation and contract are unchanged.

## Unmodified reference discrepancies

| Width / raw bit patterns | Original classical result | Original executed result | Mathematical signed `a>b` |
| --- | --- | --- | --- |
| 1 / a=1, b=0 | 1 | 1 | 0 (-1 > 0) |
| 2 / a=3, b=0 | 1 | 0 | 0 (-1 > 0) |
| 2 / a=2, b=1 | 1 | 1 | 0 (-2 > 1; subtraction overflows) |
| 3 / a=3, b=4 | 0 | 0 | 1 (3 > -4; subtraction overflows) |

These are tests against the original `on_classical_vals` and
`decompose_bloq().call_classically`, without modifying the reference. The
signature uses `QUInt` even when `signed=True`; the bit-pattern interpretation
above is explicit. No upstream defect is silently corrected.

## Validation and resource evidence

`test_signed_linear_greater.py` compares the full original forward and adjoint
primitive dependencies, operands and feedback at widths 1/2/3/4/5/8.
All input patterns and both target states are checked through width five;
width eight uses named boundary samples. A compositional channel certificate
checks every erase block's zero-work invariant and all measurement histories.
Widths 1–4 also run every raw measurement branch on complete complex input
superpositions, with an entangled spectator and arbitrary target, retaining
unnormalized probabilities and phases. Two applications restore the input.
This proves parity with the executed source, including the listed defects.

| Eight-bit path | Measured T | Measured CX | Measured X | Measurements / resets | Quantum work | Total qubits | Classical slots |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Signed difference sign | 28 | 83 | 18 | 7 / 7 | 7 | 24 | 7 |
| Existing unsigned comparison | 32 | 95 | 20 | 8 / 8 | 10 | 27 | 8 |

Separately, source `QECGatesCost` estimates 28/32 T and 7/8 measurements;
`QubitCount` estimates 24/27 total qubits. The source logical call graph uses
41/47 CNOT and 7/8 And-compute/erase pairs. These logical counts must not be
confused with the expanded primitive CX counts above. There is no saving claim
between equal mathematical contracts: the signed source has the overflow
limitation. Source estimates and primitive counts are tested independently.

Run the native user workflow (no Qualtran dependency):

```bash
python -m examples.algorithms.register_signed_difference_sign
```

The example uses `circuit.append(DifferenceSign())` on signed values -1/0/1,
where three-bit subtraction cannot overflow. It checks complex amplitudes,
clean work, arbitrary target restoration and prints measured primitives and
separate estimates. No stable root API is added.

Local acceptance: 31 new tests and 2,012 combined register/field/dataflow
regressions pass without skips. The shared comparison module has 100% statement
coverage. Strict mypy passes for all 962 production files and the example;
minimum/current Black, Ruff, scope, architecture and multi-level IR checks pass.
The reference comparison file's Git blob matches the immutable source inventory.
