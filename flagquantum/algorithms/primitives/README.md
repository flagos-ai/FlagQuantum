# Algorithms primitives

Reusable quantum primitives that the algorithm modules share. This layer owns
the quantum Fourier transform, phase estimation, state preparation, and oracle
synthesis, and all four of them ship today.

A primitive is admitted here on three dimensions, and
`contracts/primitives-admission-contract.toml` is the measured record of all
three for every export.

**The consumer fixes the ground.** A primitive is admitted when more than one
algorithm module needs it or is expected to need it and the expectation is
confirmed, or when it is a public unit callers use directly: state preparation
shipped on the public-unit ground, with no consumer inside `flagquantum/` at
all, and the Fourier transform on the expectation ground, admitted with one
consumer, phase estimation, and confirmed when amplitude estimation landed. This
package is not a general-purpose quantum toolkit, and a construction only one
workflow uses stays in that workflow's module until admitting it is justified,
whether by a second consumer arriving or by a grounded expectation of one.

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

- `qft.py`: the quantum Fourier transform and its inverse, emitted as a circuit
  fragment that callers append to a circuit they already hold, and the primitive
  phase estimation consumes.
- `phase_estimation.py`: phase estimation over a controlled unitary, with the
  resolution and the success bound the counting register buys.
- `state_preparation.py`: a uniform superposition, and an arbitrary state built
  from a classical amplitude vector by uniformly controlled rotations.
- `oracle.py`: the reversible classical building blocks the oracle units are
  composed from -- a multi-controlled X and a bit-string comparator -- and the
  truth-table synthesis of a phase or bit oracle on top of them.
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
