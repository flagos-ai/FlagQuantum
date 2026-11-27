# Recursive gradient Fourier constructor disposition

## Complete source and ownership

The complete qft/qft_phase_gradient.py, its original tests and the direct
multiplication, Hadamard, swap and resolver dependencies remain frozen at
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Six exact Git blobs
and the actual locked Kickmix 0.1 Simulator are freshly verified in
`g2-gradient-constructor-intake-20261123.json`.

| Source definition | Native owner |
|---|---|
| QFTPhaseGradient bitsize and with_reverse | Compiler formats/legacy_gradient_qft_factories.py retains integer metadata, serialized Boolean reversal and the original default. |
| signature | Existing register types retain MSB-first unsigned q and unsigned fractional phase_grad through words of equal width. Both remain caller-owned. |
| decompose_from_registers | Existing phase/gradient_qft_component.py recurses on the first half without reversal, accumulates the reversed first-half product into the full gradient, recurses on the second half without reversal, then optionally swaps the outer q word. |
| PlusEqualProduct dependency | Existing arithmetic/product_accumulation_component.py owns the complete native helper closure, carry allocation, measurement feedback and erasure. The source product leaf has an arithmetic operator and model without a primitive decomposition. |
| build_call_graph | Existing Algorithms qft_phase_gradient_resource_model.py retains the original concrete aggregate and symbolic logarithmic bound. |
| _qft_phase_gradient_small and DocSpec | The unchanged original example resolves through the stored wire at library depths zero, one and two. |

The format owner supplies no numerical kernel or independent model. The native
component and its allocation events remain identical through the existing
binding and circuit.append(op) entry. Shared products reuse and return their
private clean work. The live gradient is preserved only when supplied in the
appropriate eigenstate. General gradient inputs retain the joint source
operator.

## Evidence and separate accounting

Constructor checks cover widths one through five with both reversal settings,
source top-level calls and operand labels, all helper definitions, complete
bound streams and output ownership. Widths one through three additionally
compare the complete joint source operator. Existing affected Fourier checks
supply noncontiguous bindings, coherent spectators, every applicable small raw
measurement history, exact inverse recovery and zero work. Their unchanged
Program equality is verified before reusing those execution receipts.

The three-bit reversed constructor emits 24 T/T-dagger, 46 CX and six
measurements and resets on ten physical wires. Its separate source aggregate
model estimates 40 T-equivalent and six qubits. At width four the existing
native owner emits 52 T-family gates and 13 measurements, while executed source
product-leaf models sum to 48 T and the source aggregate estimates 40 T. Model
values, primitive measurements and physical storage remain separate quantities.

## Retained original boundaries

The original aggregate dictionary collapses equal child keys at even widths.
Its recursive child declarations default to reversal while executed children
use false. The original automatic general count check remains an expected
failure. These declarations remain queryable through their existing model
owner. The product leaf supplies no comparable original primitive stream.

Original zero-width decomposition raises KeyError for q. Symbolic width,
including direct Sympy Integer, refuses decomposition. The constructor retains
bodyless declarations for these direct cases, and physical binding refuses.
The original wire normalizes Sympy Integer to Python int and therefore resolves
a concrete body. Unapproved signed storage and fully symbolic physical execution
remain outside this file's acceptance.

## Acceptance receipts

The concentrated Fourier and constructor regression passes 361 cases. The local
integration scope passes 145 cases. The unchanged original suite passes 14 with
one expected source failure. Strict checking covers both changed Python files.
Black, Ruff, Compiler scope, language, architecture, multilevel IR and public
API checks pass. Dependency checking retains the existing 720 findings.
Receipts use prefix g2-gradient-constructor and date 20261123.

## Faithful whole-file disposition, November 28

Q-0640 is eligible for migration acceptance under the approved
[G2 closure rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md). Complete source
recursion, stored constructors and catalog dependencies already have native
owners; the current factory delegates to the same typed component. Published
`d3ec41d32` and existing full operator certificates are reused.

Equal-child key collapse and incorrect modeled child reversal remain upstream
model-correction follow-ups owned by the gradient-QFT resource model. The
executed recursive graph, aggregate declaration and actual primitive inventory
remain checked separately. PlusEqualProduct has no original primitive
decomposition: its arithmetic operator certifies the joint channel, rather
than a nonexistent source primitive recipe. Neither defect is a missing native
execution path. Separate QPE and THC consumer responsibilities remain open.

Final verification for this disposition: 336 concentrated cases, 217 independent
Cirq 1.6 cases and 248 local integration cases pass. The three unchanged original
test files report 61 passes and four original expected failures. Six unchanged
Compiler owners pass strict typing, format and lint checks. Architecture, IR,
public API, required checks, Cirq contract, scope, English, links and whitespace
checks pass. The existing 570 dependency findings remain; none belongs to the
changed audit paths. Eight source/test blobs are verified by the mandatory
source-acceptance fixture, which also verifies Qualtran 0.7.0 and the real
Kickmix 0.1 Simulator. No source file, implementation or test was changed by
this acceptance batch. Receipts use prefix g2-fourier-source-closure.
