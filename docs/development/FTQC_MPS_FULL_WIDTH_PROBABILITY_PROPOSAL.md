# Proposed bounded full-width MPS probability route

Status: proposal only. Approval or merge of this document does not authorize
implementation, production activation, numerical validation, or further research
execution.

## Evidence and the decision

The admitted source is main `19b9e6ab727d5a82432d8283dc9790235c9ab7e0`.
The development-base audit of main
`3ab96490d5de080e48f909cbceee517cb74db04b` found the relevant native MPS,
measurement, observer and marginal-test files unchanged.
The single authorized diagnostic stopped after one probe and one MPS worker;
its new producer retained two negative values, approximately -5.55e-17 and
-8.33e-17. Shape, CPU placement, float64 and finiteness passed. The new total is
8 starts out of the 161-start ceiling, with no restart authority remaining.
The original failed producer did not retain its values. These new values cannot
be attributed retroactively to that producer.

Runtime's parity-reconstruction documentation acknowledges small negative
floating-point entries. The study independently requires nonnegative outputs.
Neither clipping nor a weaker observer is proposed. The exact internal arithmetic
that produced this particular diagnostic has not been traced.

The proposed design reuses the existing native
[`MPSState.amplitude(bitstring)`](../../flagquantum/simulation/mps/state.py)
contraction for a strictly bounded request for every wire. General
partial-marginal projector norms, QR sweeps and their rank-deficient gradient
questions are outside this proposal. The existing
[Runtime measurement implementation](../../flagquantum/runtime/measurements.py)
and [marginal boundary tests](../../tests/unit/test_measurements_marginal_reduction.py)
remain authoritative until an implementation-specific review approves a change.

## Proposed scope for a future implementation approval

Only a noise-free exact native MPS output, CPU complex128, batch size one,
one through three qubits, with a probability request naming every wire exactly
once, is eligible. Both natural and permuted full-wire order must be supported.
The native exact configuration must retain cutoff zero, unlimited bond dimension
and zero reported truncation. No approximate MPS, noise, GPU, arbitrary target,
partial marginal, sample/count request, mixed basis or tensor-network route is
included. The new internal helper must decline ineligible inputs before any
amplitude contraction, leaving the existing measurement path reachable.

Simulation owns the calculation. Runtime would delegate only the eligible
analytic probability request to a private Simulation helper using its existing
native MPS type boundary. No new public method, export, option, registry,
serialized field or pickle identity is proposed. Existing public
`MPSState.amplitude` is reused without changing its signature or implementation.
The bounded route must not silently become a generally selected strategy.

## Calculation and invariants

For each outcome in the requested wire order, map its bits back to physical
wire order and invoke the existing single-amplitude contraction. Compute the
real Born weight as real-part squared plus imaginary-part squared. Accumulate
only the requested real probability vector, not a complex statevector. Never
call `to_statevector`, `state`, `probabilities`, `canonicalize`, batched
`orthogonalize_left`, a statevector executor or a tensor-network fallback.

For finite amplitudes, addition of these two nonnegative squares yields a
nonnegative floating-point weight. Overflow or any nonfinite result is an error,
not a value to clamp or normalize. The existing observer retains its strict
nonnegative guard and its approved atol=1e-12, rtol=0 comparison/normalization
checks. No normalization is introduced in this helper.

The existing contraction uses one MPS bond environment per amplitude and does
not contract an unselected physical dimension into a dense state. An all-wire
request already asks for 2^n real outputs. This proposal permits only n<=3 and
at most eight existing amplitude calls; it does not justify an unbounded output
allocation or a partial-marginal implementation with 2^n hidden work.

No tensor in the result MPS is overwritten. Configuration, truncation records,
route metadata, device and dtype must remain unchanged. Existing differentiation
through amplitude contraction must be preserved: no detach, NumPy conversion or
new factorization is allowed. This requirement is a future validation gate,
not a claim that gradients have been tested here.

## Validation required only after implementation-specific approval

- Fixed independent analytic cases with zero support, Bell correlations and
  phase-sensitive weights; permuted wire-order mappings and batch shape.
- Small independent statevector references for the approved domain with the
  original atol=1e-12 and rtol=0; nonnegativity tested separately and strictly.
- Test doubles that reject every dense-state/fallback/normalization operation;
  at most 2^n single-amplitude calls and no mutation of input tensors or metadata.
- Eligibility boundaries: partial wires, n>3, other batch sizes, dtype/device,
  approximation/noise and non-native duck-typed targets keep the existing route.
- Finite-value failure and original observer guard/diagnostic preservation.
- Differentiation against an independent analytic derivative within the same
  bounded domain, with no QR or rank-threshold workaround.

No such numerical tests have run. Existing tests and frozen fixtures must not be
rewritten to make the route pass. If an original test protects all non-dense
measurement routes without distinguishing this explicitly requested full-width
bounded route, identify that conflict for review before changing the assertion.

## Separate approval boundaries

This change contains only this proposal and a factual
[master ledger](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md#fts-10d-bounded-phaseoverlap-cost-decision)
update. Approval or merge of this documentation does not activate the numerical
route. All current public outputs and production paths remain unchanged.

Implementation, production route selection and the above numerical tests need
separate explicit scope approval. Any later study probe, single-point retry or
campaign restart also needs its own amendment. No hidden qualification workers
are included. Old and new failed raw trees remain immutable.

## Existing public user example for a documentation PR

CPU core installation: Python 3.12, a supported PyTorch dependency resolved by
`python -m pip install .`; no optional SDK or GPU is required. Save this code as
`bell_probabilities.py` and run `python bell_probabilities.py` from outside the
checkout using that installed interpreter.

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1).t(0).h(0).h(1)
for mode in ("statevector", "mps", "tensor_network"):
    result = fq.run(
        circuit,
        outputs=fq.probabilities([0, 1]),
        options=fq.ExecutionOptions(
            mode=mode,
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
        ),
    )
    values = result.probabilities
    assert values is not None and values.shape == (1, 4)
    print(mode, *[f"{float(p):.8f}" for p in values[0]])
```

Expected stdout with the existing admitted implementation:

```text
statevector 0.42677670 0.07322330 0.07322330 0.42677670
mps 0.42677670 0.07322330 0.07322330 0.42677670
tensor_network 0.42677670 0.07322330 0.07322330 0.42677670
```

This Bell example demonstrates existing functionality. Final reviewed-head
execution evidence belongs in the PR description. It does not demonstrate the
proposed calculation or replace a failed research point.

## Completion claim

At most this work can resolve a narrowly described output-construction contract.
It does not finish the finite cost study, choose a generally faster method,
certify capacity, close FTS-10 or close M2. R01-R15/FTS-02/09/10/11 remain partial.
