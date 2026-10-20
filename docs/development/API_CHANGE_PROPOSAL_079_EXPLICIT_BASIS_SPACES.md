# API Change Proposal 079: Ordered explicit bases for constrained-space review

## Status and user outcome

**Proposed; not approved for production implementation.** This is the next bounded
M1 representation review after piecewise-constant controls. It preserves CircuitIR
1.0 and all accepted behavior of current product-space requests. It proposes an
explicit labelled-basis path on the existing Lindblad API and a new plan format
for that path only; these changes require owner approval before implementation.
No public constrained-space support follows from this document or its numerical
reference. Existing plans 1.3/1.4/1.5 remain byte-identical.

A researcher working in a fixed-number sector or a fixed-charge fusion basis
needs a stable association between matrix axes and allowed basis labels. Equal
matrix dimensions do not establish equal tensor factors, physical systems or
allowed operations. The concrete outcome is to retain declared basis labels
through validation, planning, execution, results and JSON replay, while using the
existing numerical path. An internal immutable basis value supports that concrete
consumer. This is smaller than a fusion-program IR, a physics model builder or a
new namespace.

## Reuse and scope

| Existing owner | Responsibility retained |
| --- | --- |
| `core/ir/state_space.py` | Product basis order and internal mathematical state-space data. |
| `core/parameters.py` | Scalar parameter traversal and binding. |
| `observables/boson.py` and operator matrices | Existing operator construction and bounded conversion. |
| `simulation/lindblad_request.py` | Existing numeric shape, state, Hamiltonian and collapse checks. |
| `simulation/lindblad_generator.py` | Liouvillian and its shared operator algebra. |
| `simulation/lindblad_integrators.py` | All three existing interval methods. |
| `lindblad/_plan.py` and EvolutionResult | Existing encoded numeric reference execution and outputs. |

The existing product-space candidate requires positive tensor-factor dimensions
and explicitly refuses fusion descriptors. It cannot express these ordered
non-factorized labels without erasing that distinction. Any approved contract
delta must reference that existing baseline and retain its refusals; it must not
create a second baseline, policy checker or capability registry.

No new package, numerical engine, dispatcher, parameter language, training loop,
representation registry or base-class hierarchy is proposed. Production Core
continues to depend only on the standard library for this value.

## Proposed existing-API extension

Append optional keyword-only `basis_labels=None` after `parameter_values` on the
same six entries extended by proposal 073: `flagquantum.lindblad.plan`, `run`,
`adjoint_gradient`, the two low-level Simulation planning/evolution entries, and
Simulation's initial-density adjoint. Existing parameter order/defaults stay
unchanged. No root or namespace export is added.
The appended parameter type is
`basis_labels: Sequence[Sequence[int | str]] | None = None`.

`basis_labels` describes the ordered basis of the supplied matrices. It is
mutually exclusive with explicit `dimensions` or `n_qubits`, even when their
numbers agree. It has at least one label and follows the exact structure rules
below. It never asks the framework to infer a tensor product or an embedding.

Only full D-dimensional numeric states and D-by-D dense operator matrices are
accepted on this path. Existing full-space term lists and scalar parameter
bindings, including proposal 073 schedules, remain available. Pauli descriptors,
bitstrings, named qubit collapse operators, local selectors and direct boson
conversion are refused: those forms require a declared product factorization.
The model may explicitly construct a bounded ambient boson matrix using the
existing API and validate its projection before supplying a reduced matrix.
There is no automatic projection or silent removal of leakage.

The proposed spelling below is **proposed, not implemented**:

```python
# Future API, executable only after approval and implementation.
labels = (("1", "sigma", "1"), ("psi", "sigma", "1"))
plan = fql.plan(H_in_this_basis, [1., 0.], [0., 1.], basis_labels=labels)
result = fql.run(plan)
assert plan.basis_labels is not None
index = plan.basis_labels.index(("psi", "sigma", "1"))
probability = result.populations[-1, index]
```

The user keeps their model's fusion convention and total-charge meaning; the
framework preserves these literal labels, not a certified physical fusion model.
The existing population/density/batch axes and EvolutionResult type remain
unchanged. EvolutionResult.to_dict remains unchanged and carries plan_identity,
not a second copy of basis data; retain the plan JSON to interpret a stored
result dictionary. Add a read-only `LindbladPlan.basis_labels` property returning the
normalized nested tuple for this path and `None` for existing product plans.
Closed plans refuse a non-None basis_labels override in `run(plan, ...)`.
For labelled LindbladPlans, existing `n_qubits` and `dimensions` accessors raise
CapabilityError with guidance to inspect basis_labels; their annotated return
types and all existing product-space return values stay unchanged. Do not report
a label count as a count of qubits or tensor factors. EvolutionPlan.n_qubits
also raises for this path; its raw dimensions field remains None.

| Invalid request | Public refusal |
| --- | --- |
| Malformed, empty, duplicate, or inconsistent-arity labels | EvolutionValidationError, code `invalid_basis_labels`, field `basis_labels` |
| Labels with explicit dimensions or n_qubits | Existing `conflicting_state_space`, field `basis_labels` |
| Qubit bitstrings, Pauli inputs or local selectors | Existing `qubit_only_input`, with the original input field |
| Direct BosonOperator without product dimensions | Existing `boson_dimensions_required` |
| Operator/state shape does not match label count | Existing shape/state validation with input context |
| Malformed structural labels, schema or inconsistent plan metadata | Existing SerializationError category |
| Decoded numeric Hamiltonian/state rejected during plan revalidation | Existing EvolutionValidationError category |

All knowable refusals precede interval integration. Labels are shared across all
initial-state batches and control intervals; no label-batch axis or dynamic basis
change is inferred. The returned mathematical request has an explicit basis,
not a certificate that its matrices preserve a declared physical charge.

## Proposed internal contract

Add `ExplicitBasisSpace` beside `ProductStateSpace` in the existing internal
`flagquantum.core.ir.state_space` module, after this contract is approved and its
fake/conformance baseline lands. The existing state-space resolver consumes it;
no new registry is added. This is a proposed spelling, not an import users can execute today:

```python
# Proposed internal API only; not available in the current package.
ExplicitBasisSpace(labels: Sequence[Sequence[int | str]])
# .labels -> tuple[tuple[int | str, ...], ...]
# .dimension -> int
# .is_qubit_space -> False
# .basis_index(label: Sequence[int | str]) -> int
```

The labels name a finite orthonormal basis for the supplied amplitudes and
operators. No Gram matrix, non-orthonormal frame, continuous basis or implicit
change of metric is accepted. The model owns that orthonormal-basis convention.

The value owns exactly the ordered labels and their indexing:

- The outer and inner inputs must be finite ordered sequences, excluding strings,
  bytes, mappings and generators. Copy them to nested immutable tuples.
- Require at least one nonempty label, common label arity, and no duplicates.
  Atoms are exact builtin int values (including signed labels) or nonempty
  exact builtin str values; subclasses are not implicit identifiers;
  booleans, floats, tensors, nested containers and arbitrary objects are refused.
  Labels are identifiers, not inferred Fock occupations or fusion rules.
  Common label arity is a lexical structure, not a subsystem count.
- Preserve caller order and exact atom values. Do not sort, strip, case-fold,
  coerce numeric strings or infer aliases. `dimension` is the number of labels.
- `basis_index` applies the same label validation, then returns its unique index.
  Malformed values and unknown labels raise ValueError; a non-sequence input
  raises TypeError. Construction uses the same distinction.
- The object is frozen, including a defensive copy of mutable input sequences.
  Equality and hashing compare the type and ordered labels.
- There is no `dimensions` field, tensor-product factorization or implicit
  embedding. `is_qubit_space` is False even for dimensions two or four.
  ProductStateSpace and its mixed-radix indexing remain unchanged.

The name describes the mathematical data this value actually owns. A generic
`ConstrainedStateSpace` would imply validation of arbitrary physical constraints;
a `FusionStateSpace` would imply a fusion model. Neither claim belongs in this
small Core value. Do not add an opaque metadata dictionary to simulate those
responsibilities.

## Domain meaning stays with the model

The first witness uses labels `(0, 1)` and `(1, 0)` for a two-mode, one-photon
sector. Its ambient mode truncations, conserved number operator and
embedding belong to that model, not the explicit-basis value.

The second witness uses `("1", "sigma", "1")` and `("psi", "sigma", "1")`
for four Ising sigma charges with total charge 1 in a left-associated fusion tree.
The model must separately retain the anyon types, total charge, tree, fusion
rules and F/R convention. Core validates label structure, not that physics.
The convention follows [Nayak et al., section II.A, equation (50), and Appendix A.1](https://arxiv.org/abs/0707.1889):
`F=[[1,1],[1,-1]]/sqrt(2)`, `R=diag(exp(-i*pi/8),exp(3i*pi/8))`, and
`B2=F.conj().T @ R @ F`. The reference checks unitarity, braid consistency and
fusion-channel probabilities within this declared convention.

A basis list alone is not a complete constrained physical model. In particular,
identical labels do not certify identical total-charge sectors or basis trees.
The proposed mathematical plan identifies its exact matrices and literal labels,
not a domain-specific anyon model. A later native anyon program must retain its
typed model context and sector operations in its own reviewed lowering identity.
This slice closes only explicitly labelled matrix-space execution; it does not
close the M3 anyon-program gate.

## Plan and compatibility contract

Labelled requests alone use `flagquantum.lindblad_plan` version **1.6**, including
labelled requests with piecewise-constant coefficients. Store the normalized
ordered labels in `request.basis_labels`; derive D from their count. Keep the
current numeric Hamiltonian or ordered Hamiltonians, collapse/state/observable
payloads and numerical decisions. Do not serialize a second symbolic model or
a parameter graph.

For 1.6 only, decision metadata carries the derived `basis_labels` and
`n_wires=None`; `dimensions` is absent and request `n_qubits` is None. There is no
defensible wire count or product dimension to invent for a non-factorized
mathematical basis.
The request remains authoritative. The decision copy is generated from the same
normalized Core value and must exactly match on revalidation, following the
existing 1.4 request/decision product-dimensions pattern; no label registry or
independent basis definition is introduced. Widen EvolutionPlan's n_wires
annotation to `int | None`, preserving every existing product-space value and
serialized payload. Add an optional basis_labels field at the end, omitted from
existing payloads. The public baseline/candidate
impact must be reviewed explicitly; do not weaken the checker to admit it.

The plan's D, state/operator shapes, label order, time intervals, schedule flag and
recomputed decision must agree. Readers reject duplicate/malformed labels,
product-only fields, unknown semantic fields, false wire counts or dimension
mismatches even after rehashing. Versions 1.3/1.4/1.5 refuse labelled semantics
rather than discarding them. Equal D with different labels has a different
identity; permuting labels and all corresponding matrix axes changes coordinate
identity but preserves probabilities addressed by the same label.

Existing static qubit/product and scheduled product requests retain versions
1.3/1.4/1.5 and exact identities, optional-field omission, numeric trajectories,
Pauli fast paths and error behavior. Supplying basis_labels=None takes that
original path. Live matrices/coefficient tensors retain their graph; JSON retains
numerical values and labels, with the existing missing-gradient-graph refusal.
Initial-density adjoints reuse the same reverse interval generators and return
cotangents in the supplied coordinate order; no new control-adjoint promise is
made. No gradient is defined for discrete labels.

## Migration and release scope

This is an additive path, with no deprecation or removal. Current calls and
accepted plan 1.3/1.4/1.5 payloads remain unchanged. A caller opting into labels
uses the same matrix/state coordinates and accesses populations by a label index;
it stops pretending that a two-dimensional sector is a product qubit. Consumers
of the new plan version must read dimension and basis_labels, and must handle the
new path's nullable n_wires. Older readers must refuse the new version. A release
note and the approved candidate/API checks accompany implementation; this proposal
alone changes neither a snapshot nor a released contract.

## Runnable current-API numerical reference

Run `python -m examples.constrained_basis_reference`. It uses only APIs already
implemented; its expected output is:

```text
One-excitation transfer: 0.50000000
Control derivative: 1.00000000
Fusion-channel probability: 0.50000000
Braid relation verified: True
Constrained metadata in existing plans: False
```

The first example builds an exact invariant subspace of a four-dimensional
boson exchange Hamiltonian constructed by the existing algebra and passes its
two-by-two matrix to the existing solver.
For that example the analytic transfer law is `sin(theta)**2`; at `theta=pi/4`
the probability and derivative are 0.5 and 1.0. Check the full-space evolution
and the omitted-block coupling independently; projecting a Hamiltonian that
leaves the sector would erase real leakage and is not an admitted lowering.
For an open model, each collapse operator must also preserve the chosen sector,
or explicit loss states must be retained; Hamiltonian invariance alone is not
sufficient. This reference includes no collapse operators.

The second supplies a two-by-two Hamiltonian whose exponential realizes B2 as a
numerical witness. Current `dimensions=(2,)` plans describe an encoded qubit,
not a fusion space, and do not retain the caller's fusion labels or sector.
Density results cannot certify an overall unitary phase. Direct matrix checks
of the declared F/R convention are separate from that density-matrix comparison.
No native braid instruction, fusion measurement, decoder, microscopic Majorana
model or universal gate-set claim follows.

## Reference validation evidence

Against the post-#647 baseline `4f9d5a6c010b4b351579bfe1e95053ba58aa09b6`,
the five current-API reference tests pass on Python 3.10, 3.11 and 3.12 with
CPU Torch complex128. The runnable example produces the exact output above.
The tests compare full and reduced densities within `1e-11`, the analytic
control derivative within `1e-10`, and the declared F/R matrices and their
logarithm witness within `1e-14`. These are precision budgets for these small
references, not global solver guarantees or evidence of the proposed API.
The actual CI strict command
`mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum`
passes all 772 existing source files; production source is unchanged.

## Approval, implementation and acceptance sequence

1. **Contract decision.** Review the internal name/structure rules, appended
   basis_labels keyword, plan 1.6, optional metadata, nullable n_wires meaning,
   product-space compatibility and explicit domain limits. The owner must
   approve these changes before dependent implementation.
2. **Integration baseline after approval.** Add a narrow versioned candidate,
   a test-only fake and conformance consumers through existing plan/run, following
   proposals 078 and 073. Preserve the old product/fusion refusal test and existing
   schema fixtures. Do not edit protected snapshots or policy to force a pass.
3. **Core and Simulation replacement.** Add the small immutable basis value in
   the existing module; extend the current resolver, normalizers, planner,
   trajectory and reverse-adjoint path. The same conformance assertions run
   against the fake and real entry points. No new solver/loop/factory is accepted.
4. **Numerical and physical bounds.** Validate full-space/reduced-space agreement
   for an invariant excitation sector, analytic gradients and basis-permutation
   covariance; include a non-invariant projection counterexample. Check F/R
   unitarity, the braid relation and independently calculated fusion probabilities
   with dtype/method-specific accuracy bounds. Model context and the mathematical
   plan's guarantees remain distinct.
5. **Replay and compatibility.** Preserve numeric values and ordered labels
   through JSON; test live gradients and an actual optimizer update, shared
   initial-state batches, all three methods and initial-density adjoints. Reject old-schema disguised semantics,
   forged label/count/shape fields, semantic overrides and known-unsupported
   qubit/product forms before integration. Test unchanged 1.3/1.4/1.5 identities
   and CircuitIR 1.0; a two-state labelled basis must stay non-product.
6. **Repository gates.** Run strict mypy with the actual CI flags, style,
   architecture, dependency, ownership, API/IR/documentation checks,
   supported-Python focused cases and required smoke/unit plus integration tiers.
   Update authoritative capability/parity sources and generated views only after
   the public path passes these gates; a preparatory example is not that evidence.

## Alternatives and explicit non-goals

Keeping only a matrix dimension loses label order and silently suggests product
semantics. Reusing `ProductStateSpace((D,))` as the constrained representation
would retain that ambiguity. A generic state-space hierarchy, fusion-rule
registry, solver, event language or domain package would exceed both verified
consumers. The proposed labelled-matrix slice reuses the current execution path while leaving future native physical-model contracts explicit.
