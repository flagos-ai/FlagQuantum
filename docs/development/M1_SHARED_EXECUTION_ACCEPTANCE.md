# M1 shared mathematical execution: acceptance review

## Decision requested

Review the implemented, bounded CPU mathematical scope of M1 against the
[roadmap](../roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md). The implementations and
their approved contracts have merged; this document consolidates their evidence
for milestone acceptance. It does not approve a new API or declare M2-M4 complete.
The capability catalog and parity contract remain the authorities for support
and maturity; neither is changed by this review.

The reference revision is main `4488cc5a17abb631ebc1c4d46d698bb0c408aa1e`.
The observations below were reproduced on 2026-10-20. Later changes must retain
these acceptance properties rather than inherit success from a different tree.

## Requirement-to-evidence map

| Roadmap requirement | Implemented consumer | Independent acceptance |
| --- | --- | --- |
| Mixed dimensions without qubit padding | Three-level leakage and spin-mode exchange use the existing Lindblad namespace with explicit product dimensions | `test_lindblad_mixed_dimensions.py`: analytic Rabi and decay laws, basis order, time refinement, motional cutoff, batching and gradients |
| Two physical models share arithmetic and results | Both models use the existing request resolver, Liouvillian, three integrators, trajectory and EvolutionResult | The same public plan/run path is compared to analytic laws and independent dense references; no route-specific solver is introduced |
| Existing boson algebra reaches execution | Direct BosonOperator Hamiltonians, jumps and low-level observables reuse materialization with explicit truncations | `test_lindblad_boson_consumer.py`: two-photon interference, loss, spectator ordering, exact two-excitation cutoff and its insufficient-cutoff counterexample |
| Trainable coefficients reuse common binding | Static operator sums bind existing Parameter and ParameterExpression values | `test_lindblad_operator_coefficients.py`: analytic derivatives, complex conjugate terms, dense-plan equivalence, initial-density adjoints and an actual optimizer update |
| Bounded time-dependent controls | One parameter mapping per interval drives the existing trajectory and all three integrators | `test_lindblad_piecewise_controls.py` and `test_piecewise_control_contract.py`: noncommuting dissipative references, refinement, finite differences, optimizer updates and reverse adjoints |
| Constrained-space design includes a non-product consumer | Ordered explicit bases represent a photon sector and a declared fixed-charge Ising matrix reference | `test_explicit_basis_contract.py` replaces the fake with real factories; `test_lindblad_explicit_basis.py` checks vacuum-inclusive loss, coordinate permutations and discrete adjoints |
| Unsupported requests fail before integration | Conflicting spaces, malformed labels, unsupported qubit forms and invalid interval data are refused | Existing public refusal tests retain their error categories and input context; a two-state explicit basis remains non-product |
| Existing behavior and replay are preserved | Static product plans remain 1.3/1.4, scheduled product plans 1.5; labelled plans use 1.6; CircuitIR stays 1.0 | Fixed old payloads, identities and omission rules remain tested; static/scheduled live and numeric replay are checked separately |

The named test modules are under `tests/unit/`. Their names identify the
scenarios rather than a second capability registry. The product-space and
explicit-basis values remain in the existing Core IR module; they do not create
a physical-model hierarchy, hardware package, binder or numerical engine.

The M0 precision repair is retained in
`test_algorithms_chemistry_integrals.py`, including the analytic cancellation
reference. Matrix-free closure and first/second derivative checks remain in
`test_matrix_free_closed_subspace_gradients.py`.

## Reproduce the user workflows

From the repository root, with the normal development environment installed:

```bash
python -m examples.mixed_dimension_dynamics
```

Observed Python 3.12 CPU output:

```text
Three-level leakage probability: 0.11757891
Coupling derivative: 0.32210884
Spin-mode dimensions: (2, 3)
Probability of |spin=0, occupation=1>: 0.11757891
```

The two models deliberately reach the same execution path. The example owns
their Hamiltonians and interpretation; Core owns their finite product spaces.
Couplings and collapse rates use the inverse of the declared time-grid unit.

The other existing examples were also run successfully on that revision:

| Command | Observed result and meaning |
| --- | --- |
| `python -m examples.boson_interference` | Without loss, P(0,2) and P(2,0) are 0.5 and P(1,1) is zero. At loss rate 0.2 per mode, the two-photon survival is 0.63839443. The model retains loss outcomes. |
| `python -m examples.train_boson_coupling` | Population loss falls from 0.01158038 to 0.00001147. The fitted coupling is 0.68787697 against a reference 0.7; this demonstrates optimization, not an exact fit or hardware calibration. |
| `python -m examples.train_piecewise_drive` | Initial probability 0.25121448, gradient [0.04337116, 0.08674232, 0.02168558], and `Training reduced loss: True`. |
| `python -m examples.constrained_basis_reference` | Transfer 0.5, control derivative 1.0, fusion-channel probability 0.5, braid relation verified and declared basis labels retained. |

These commands reuse the existing example sources rather than copy their model
builders into another framework. Printed decimal values are observations of
these small references; the tests own their numerical error budgets.

## Verification record

The following focused selection passed **775 cases separately on Python 3.10,
3.11 and 3.12**, using CPU PyTorch:

```bash
python -m pytest -q \
  tests/unit/test_lindblad*.py \
  tests/unit/test_matrix_free*.py \
  tests/unit/test_product_state_space_contract.py \
  tests/unit/test_explicit_basis_contract.py \
  tests/unit/test_piecewise_control_contract.py \
  tests/unit/test_constrained_basis_reference.py \
  tests/unit/test_algorithms_chemistry_integrals.py
```

This selection includes the existing refusal, compatibility and replacement
assertions. No tolerance or assertion was changed for this review. The complete
local default tier also passed: **13,786 passed, 175 skipped** in 632.42 seconds
on Python 3.10. Strict mypy passed for 772 source files and all 11 repository
policy, documentation and API checks passed. The focused selection is not a
substitute for those gates or the review PR's complete remote CI/integration
workflows.

On the reference main revision, all **41 check runs** succeeded. The complete
[CI workflow](https://github.com/flagos-ai/FlagQuantum/actions/runs/39439204798),
[pre-commit workflow](https://github.com/flagos-ai/FlagQuantum/actions/runs/39439204732)
and [Local GPU Gate workflow](https://github.com/flagos-ai/FlagQuantum/actions/runs/39439205012)
all completed successfully. CI complete covered the 21 other CI jobs. Those are
integration results for the reference revision; this mathematical execution
slice's supported target remains CPU.

The implementation history is independently traceable. Each latest PR head in
this table passed all 36 checks and both complete CI/pre-commit workflows:

| Slice | Merged implementation | Complete CI run |
| --- | --- | --- |
| M0 baseline | [#637](https://github.com/flagos-ai/FlagQuantum/pull/637) | [39226271659](https://github.com/flagos-ai/FlagQuantum/actions/runs/39226271659) |
| Mixed product spaces | [#639](https://github.com/flagos-ai/FlagQuantum/pull/639) | [39237080168](https://github.com/flagos-ai/FlagQuantum/actions/runs/39237080168) |
| Boson consumer | [#640](https://github.com/flagos-ai/FlagQuantum/pull/640) | [39246090254](https://github.com/flagos-ai/FlagQuantum/actions/runs/39246090254) |
| Static coefficients | [#641](https://github.com/flagos-ai/FlagQuantum/pull/641) | [39253559348](https://github.com/flagos-ai/FlagQuantum/actions/runs/39253559348) |
| Piecewise controls | [#647](https://github.com/flagos-ai/FlagQuantum/pull/647) | [39371660057](https://github.com/flagos-ai/FlagQuantum/actions/runs/39371660057) |
| Explicit bases | [#652](https://github.com/flagos-ai/FlagQuantum/pull/652) | [39391425782](https://github.com/flagos-ai/FlagQuantum/actions/runs/39391425782) |
| Repeatable CN replay | [#656](https://github.com/flagos-ai/FlagQuantum/pull/656) | [39401240114](https://github.com/flagos-ai/FlagQuantum/actions/runs/39401240114) |

## Numerical and physical boundaries

Acceptance separates time-discretization error, iterative-solve residual,
Hilbert-space truncation and gradient error. Their existing budgets remain
independent. The [explicit-basis proposal](API_CHANGE_PROPOSAL_079_EXPLICIT_BASIS_SPACES.md)
records the CN grid-refinement evidence, independent adjoint references and
existing Krylov-autograd conditioning limits. Repeatable replay does not remove
those conditioning limits or promise identical bits across PyTorch versions
and operating systems.

JSON retains numeric values, schedule order and literal labels. It does not
restore an autograd graph. Live training rebuilds a plan after an optimizer
update; exact-plan execution refuses semantic overrides.

The model must justify its truncation or invariant subspace. A projected model
must retain real leakage/loss states or demonstrate the required invariant
blocks for its Hamiltonian and jumps. Literal labels do not certify fusion
rules, charge, orthonormality or a physical device.

M1's time dependence is the approved piecewise-constant contract. Continuous
callables/interpolation, time-dependent collapse operators and boundary-time
gradients remain outside it. Neutral-atom motion/detection events, native optical
programs, a scalable Fock backend, native braid/fusion instructions and hardware
execution retain their M3/M4 gates. Ising matrix references do not establish a
universal topological gate set.

## M2 reuse audit and next contract

The existing namespace already consumes the common ExecutionOptions,
Parameter/ParameterExpression binding and PyTorch lifecycle. Its sealed plans,
results, validation and numeric serialization have owners. M2 should compose
those owners rather than add another trainer, binder, backend registry or
result envelope.

A concrete output-request gap remains. OutputRequest expectation inputs are
currently limited to the Pauli Observable type. The following current-API
diagnostic produces the stated refusal:

```python
import torch
import flagquantum as fq

try:
    fq.expectation(torch.eye(3, dtype=torch.complex128))
except TypeError as error:
    print(type(error).__name__ + ": " + str(error))
```

```text
TypeError: expectation output requires an Observable
```

The lower-level Lindblad observable normalization already accepts full-space
Hermitian matrices with shape, finiteness and Hermiticity validation. This is
the arithmetic to reuse. A qubit Pauli decomposition or padding would lose the
non-product semantics just established by M1.

The output-name contract also needs a refusal decision. Currently, an explicit
name `expectation_1` followed by an unnamed expectation at index 1 produces the
same dictionary key and retains only the latter output. Duplicate explicit
names are already refused, but generated-name collisions are not. The M2
contract must make this observed loss explicit and decide its behavior before
changing validation; this review does not silently change existing exceptions.

The next bounded M2 deliverable should therefore be a reviewed contract for
full-space expectation requests through the existing output boundary. It must
address product and labelled coordinates, output/batch/time axes, named results,
live gradients, sealed-plan identity, numeric replay, old payload compatibility
and explicit refusal by unsupported circuit/execution paths. A replacement
baseline must precede domain implementation. This review does not authorize
widening OutputRequest, changing its serialized form or adding a new factory.
The precise API and compatibility choice belongs in that separate proposal.
