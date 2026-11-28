# Algorithms primitives

Reusable quantum primitives that the algorithm modules share. This layer owns
the quantum Fourier transform, phase estimation, state preparation, oracle
synthesis, and block encoding, and all five of them ship today.

A primitive is admitted here on three dimensions, and
`contracts/primitives-admission-contract.toml` is the measured record of all
three for every export.

**The consumer fixes the ground.** A primitive is admitted when more than one
algorithm module needs it or is expected to need it and the expectation is
confirmed, or when it is a public unit callers use directly: state preparation
shipped on the public-unit ground, and the Fourier transform on the expectation
ground, admitted with one consumer, phase estimation, and confirmed when
amplitude estimation landed. State preparation no longer stands on the
public-unit ground alone, because `algorithms/data_encoding.py` now calls it for
its amplitude half, and block encoding was admitted on the public-unit ground
after the singular-value unit held it privately and the comment recording that
promotion condition named this module. This package is not a general-purpose
quantum toolkit, and a construction only one workflow uses stays in that
workflow's module until admitting it is justified, whether by a second consumer
arriving or by a grounded expectation of one.

**No consumer anywhere is a defect, not a fourth ground.** The rule was stated
before it was measured, and the first measurement found one export — the
`StatePreparationOperator` protocol — that no module, test, or example consumed:
it rested entirely on the assumption that a third-party implementer might want
it. That assumption is no longer an admission ground. Such an export is deleted,
not grandfathered, and `retired_export` in the contract records what it was and
what to use instead. The rule has teeth only if the check refuses to record the
hypothetical as a basis, which is why the gate derives the basis from the import
graph instead of reading it from a declaration.

**Distribution semantics are claimed, not implied.** Every primitive here runs on
the default single-device fast path, and the contract records the semantic per
export by reading the same vocabulary the rest of the repository classifies
against. A primitive that names a non-default semantic would be making a
scalability claim — replicated execution reported as if it were sharding, which
rule 1 forbids — so the gate fails a module whose recorded semantic and measured
semantic disagree.

**Differentiability is recorded per export.** Two exports take a
`torch.Tensor`: `arbitrary_state` and `append_arbitrary_state`. Both read the
amplitude vector as classical data, solving rotation angles from its values and
emitting fixed angles, so no gradient flows back and the vector is detached on
entry. That is not a limitation to be discovered from a PyTorch warning; it is
stated in each docstring and recorded in the contract, with the command that
measured it. An export that claims a gradient in this table has to name an
existing test that proves it, function by function.

## Where to start

- `qft.py`: the quantum Fourier transform and its inverse, built once on its own
  register and placed onto the caller's qubits with `Circuit.compose`; the
  primitive phase estimation consumes.
- `phase_estimation.py`: phase estimation over a controlled unitary, with the
  resolution and the success bound the counting register buys.
- `state_preparation.py`: a uniform superposition, and an arbitrary state built
  from a classical amplitude vector by uniformly controlled rotations. The
  amplitude half of `algorithms/data_encoding.py` is its first consumer inside
  `flagquantum/`; that module owns the classical front end (padding and
  normalisation), and this one owns the preparation ladder.
- `oracle.py`: the reversible classical building blocks the oracle units are
  composed from -- a multi-controlled X and a bit-string comparator -- and the
  truth-table synthesis of a phase or bit oracle on top of them.
- `block_encoding.py`: the block-encoding protocol consumers are written
  against, the spectral encoding that satisfies it, and the qubitization walk
  step that encoding admits, whose eigenvalues are the arccosines of the encoded
  matrix's eigenvalues over `alpha`.
- `linear_combination.py`: a second implementation of the same two protocols, for
  a Hamiltonian given as a weighted sum of Pauli words rather than as a matrix:
  the factor is the sum of the coefficient magnitudes, stated from the data, and
  the coefficients are prepared in an index register rather than read off a dense
  matrix the construction never forms.
- `types.py`: the callable protocols the primitives are written against.
- `__init__.py`: the public primitives surface.

## Boundaries

Primitives build circuits from other circuits and from classical data. They do
not define circuit or operator semantics, compiler passes, numerical kernels,
runtime selection, provider lifecycle, or any performance claim. Numerical
kernels stay in `simulation/`, which owns gate matrices and state evolution;
a primitive calls those through the ordinary circuit API instead of
reimplementing them or carrying a second execution path.

Every primitive in this package is demonstration scale unless its own module
docstring says otherwise. Nothing here carries a capacity, throughput, or
quantum-advantage claim.

For a small change, add a focused public function, check it against an
independent reference, and run:

```bash
python -m pytest tests/unit/test_algorithms_qft.py \
  tests/unit/test_algorithms_package.py -q
python tools/check_architecture.py
```

`generalized_qsp.py` is the internal finite explicit-P/Q operation over the
existing numerical phase recursion and Compiler signal sequence. It binds
ordered named U ports and caller-owned clean workspace through
`circuit.append(operation.make_on(...))`. It has no tensor-coefficient gradient
or distribution claim. The [binding guide](../../../docs/development/GENERALIZED_QSP_OPERATION.md)
contains an executable native example, workspace obligations and source-parity
boundaries. Shape-only declarations, typed signatures and symbolic call graphs
now reuse shared owners. Physical synthesis remains its separate boundary. The
[complete source audit](../../../docs/development/GENERALIZED_QSP_FILE_AUDIT.md)
records the original precision skips and remaining admission gaps. The finite
factory and declared model are documented below.

The finite GQSP factory now composes existing grid verification, root complement
and lazy phases, with an explicit FFT provider for verification. Signal resource
calls retain source precision budgets. `generalized_qsp_model.py` independently
composes caller-priced source call counts through shared ComponentCost; it does
not infer physical or synthesized prices. The [factory/model guide](../../../docs/development/GENERALIZED_QSP_FACTORY_MODEL.md)
provides the native append example and precise acceptance boundaries.

`alias_discretization.py` owns raw source-ordered probability rounding, integer
alias donors and precision widths. Validated concrete/symbolic alias factories
reuse it; they retain their stricter input contracts. It owns no register
allocation, numerical circuit execution or sampling state. See the
[whole-source audit](../../../docs/development/ALIAS_DISCRETIZATION_SOURCE_AUDIT.md)
for original-test replacement, append consumers, bit/work ownership and separate
primitive/model counts; the register guide remains the shortest workflow.

## Chemistry lookup declarations and number phase

`chemistry_qroam_model.py` owns the clean/dirty chemistry lookup scores, the
forward/inverse block-exponent query and the one/two-axis literature models.
Its costs feed both DF and THC consumers. `chemistry_number_phase.py` binds the
most-significant system-wire phase through the existing Compiler conjunction
recipe. Ecosystem converts external objects before these native owners consume
them. Measurement cleanup belongs to the existing Compiler primitives.

```python
from flagquantum import Circuit
from flagquantum.algorithms.primitives.chemistry_number_phase import ChemistryNumberPhase
from flagquantum.algorithms.primitives.chemistry_qroam_model import chemistry_qroam_log_block_size

exponent = chemistry_qroam_log_block_size(100, 10)
phase = ChemistryNumberPhase((0, 1), 2)
circuit = Circuit(5)
circuit.append(phase.make_on((0, 1), (2, 3), (4,), (0,)))
```

The phase operation has two control wires, a two-bit system and one caller-owned
clean work wire with one measurement slot. Query tests, actual external objects
and all-branch native execution are exercised by
`tests/unit/test_chemistry_black_boxes_file.py`. Run that file together with
`test_chemistry_qroam_model.py`, `test_chemistry_symbols.py` and
`test_double_factorization_encoding_model.py` for a focused change. The complete
source owner map and measured/model separation are in
`docs/development/CHEMISTRY_BLACK_BOXES_FILE_AUDIT.md`.

## Chemistry composition ownership

Chemistry package grouping is represented by these existing domain modules.
Double factorization uses the `double_factorization_*` composition and model
owners. Hubbard preparation, selection and walk use `hubbard_prepare.py`,
`hubbard_select.py` and `hubbard_walk.py`. Standalone Majorana X/Y and spin-up
phase selection live in `hubbard_majorana.py` and `hubbard_spin_up.py`. Shared
traversal, phase conjunction
and workspace allocation remain in Compiler. External SDK objects are translated
at Ecosystem before native execution.

The license-only chemistry and DF initializers and the empty Hubbard root have
no executable behavior to duplicate. Their complete disposition and the five
Hubbard-qubitization export consumers and retained source-body gaps are recorded
in `docs/development/CHEMISTRY_PACKAGE_ENTRY_AUDIT.md`. Existing Hubbard use starts
with `HubbardWalk.from_lattice(2, 1, 2)`, binds `walk.layout` and `walk.requirements()`
through `make_on`, then uses `circuit.append(op)` on a `DynamicCircuit`. The full
runnable workflow is in `tests/unit/test_hubbard_walk.py`. Run that file with
`test_hubbard_prepare.py`, `test_hubbard_select.py` and
`test_hubbard_notebook_resources.py` when changing those owners.

Double-factorization declaration changes start in
`double_factorization_prepare_model.py` or
`double_factorization_rotation_model.py`. Preserve source signature order,
zero-width port rules and directional lookup costs. Raw signed source counts
travel through the existing component-library interchange before estimator
admission. Run `test_double_factorization_prepare_model.py`,
`test_double_factorization_rotation_model.py` and
`test_double_factorization_foreign_models.py`. The runnable model example is
`examples/algorithms/double_factorization_resources.py`. Executable changes use
the existing coefficient-driven owners and the normal append workflow covered
by `test_double_factorization_block_encoding.py`. The complete source mapping,
physical evidence and model-only boundary are in
`docs/development/DOUBLE_FACTORIZATION_PREPARE_SELECT_FILE_AUDIT.md`.

Top-level DF model changes belong in `double_factorization_encoding_model.py`.
Reuse the existing leaf owners when changing a call or its live width. Run
`test_double_factorization_encoding_model.py` and
`test_double_factorization_foreign_encoding.py` for original call/operand and
actual stored-graph consumers. Physical coefficient changes are verified by
`test_double_factorization_block_encoding.py`. Source analytical declarations
and unavailable leaf bodies remain explicit in
`docs/development/DOUBLE_FACTORIZATION_FILE_AUDIT.md`.


Standalone Hubbard changes start in `hubbard_majorana.py` or
`hubbard_spin_up.py`. Both expose a named `layout`, clean-work/feedback
`requirements()` and `make_on(ports, workspace=...)`, then use the existing
`DynamicCircuit.append(op)` path. Majorana targets use x/y/spin order, while
spin-up Z uses y/x in the second system half. Both preserve valid addresses and
restore clean work. Measurements require DynamicCircuit, and coordinates are
classical circuit configuration rather than differentiable parameters. Run
`test_hubbard_majorana.py`, `test_hubbard_spin_up.py`,
`test_hubbard_package_consumers.py`, `test_hubbard_select.py` and
`test_hubbard_walk.py` together. The file audit records source negative-control
and singleton defects and separates emitted gates from analytical declarations.

Stored Hubbard preparation starts in `hubbard_prepare_component.py`. It composes
shared uniform preparation, swaps, conjunctions and native bounded increment.
`packed_prepare_component.py` packs a supplied preparation Program through the
shared AutoPartition owner. Pass the exact inverse explicitly for an adjoint.
Selection and logical junk stay caller-owned, while child components own clean
scratch. Run `test_hubbard_stored_graph.py`, `test_packed_prepare_component.py`
and the existing Hubbard/alias/packed file tests together. Source atomic
arithmetic, singleton, symbolic and reflection-junk boundaries are recorded in
`docs/development/PACKED_PREPARE_STORED_GRAPH_AUDIT.md`.

`hubbard_majorana_component`, `hubbard_spin_up_component` and
`hubbard_select_component` retain successful complete stored Hubbard SELECT
recipes through shared Compiler traversal, partitions, swaps and exact
adjoints. Hubbard owns coordinate conventions and composition. The controlled
SELECT root retains its relative phase and all eleven original calls. See
`docs/development/HUBBARD_SELECT_STORED_GRAPH_AUDIT.md` for source boundaries,
small all-history evidence and separate literal/model resource accounts.


Factorization signatures keep raw logical precision declarations beside strict
concrete helpers. `factorization_source_formula.py` reuses concrete child models
and replaces only source-variable category/storage terms, including symbolic
comparison dispatch. It does not choose a backend or execute a statevector.
Run `tests/unit/test_factorization_source_closure.py` for constructor, exact
source-direction and model changes, then the existing SF/DF model/library tests.
Opaque source declarations do not imply a physical circuit.

`thc_chemistry_walk.source_thc_walk_cost` and `source_thc_walk_gate_cost` query
the existing THC source budgets from a preparation owner without requiring
physical Givens networks. These are model queries; literal source dependency
capture and actual emitted gates remain separate evidence.

`EllipticCurveAddition.source_call_graph()` uses the shared parameterized model
in `elliptic_curve_addition_model.py`. The root and all six source stage formulas
retain width, modulus, window, exact adjoint and encoded control identities.
`elliptic_curve_addition_parameterized_graph` also supports symbolic metadata
without constructing a physical operation. The existing label-only tutorial
graph remains unchanged. For example:

```python
from flagquantum.algorithms.primitives.elliptic_curve_addition import EllipticCurveAddition

graph = EllipticCurveAddition(3, 3).source_call_graph()
assert graph.root == ("ECAdd", 3, 3, 1)
assert len(graph.direct_calls) > 6
```

Price leaves explicitly with `graph.estimate_leaf_cost(price_by_leaf.__getitem__)`.
Unpriced modular declarations are not zero-cost operations. Reference leaf
prices in conformance tests are explicit policy inputs, not independent native
default QEC pricing or primitive instruction counts. Start with
`tests/unit/test_ecc_addition_model.py`; the ECC source audit retains the
separate physical, classical and source-defect certificates.
