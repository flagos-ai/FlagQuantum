# Explicit Pauli-sector circuit training

## Scope before implementation

Algorithms composes existing native Pauli discovery, explicit stabilizer
reduction, one-qubit circuit preparation, `fq.run` expectations and PyTorch
optimization. This is an experimental CPU `single_device_fast_path` example,
not a new solver, public contract, registry or runtime implementation.

Use the already supported three-qubit model
H = Z2 + g*Z1*X2 + 0.2*Y0 + 0.3*X1*Z2, with static g=0.4.
Discovery returns positive Y0 and X1Z2. Choose their eigenvalues +1 and -1
explicitly, then prepare the remaining qubit with RY(theta). The independent
selected-sector energy is -0.1 + cos(theta) + g*sin(theta), with minimum
-0.1 - sqrt(1+g**2). Only the float64 circuit angle is trainable.

Symbolic discovery/reduction remain static. In particular,
`lift_stabilizer_state` refuses autograd inputs; this workflow must not weaken
that boundary or detach an input silently inside an algorithm. Use its existing
static CPU path only under `torch.no_grad()` after training for bounded
full-register inspection. Gradient references construct the small physical
sector independently of production lifting. No dense eigensolver or arbitrary
state preparation is needed for optimization.

## Acceptance

- Compare native reduced energy and angle gradients with analytic formulas,
  central finite differences and independent three-qubit product states.
- Test every explicit joint sector and both signs of the static coupling.
  Replace discovered generators with supplied signed generators while keeping
  the training consumer unchanged; adjust signed eigenvalues explicitly.
- Verify optimizer updates lower energy and keep the lifted state in the chosen
  X/Y/Z joint sector, including after an update. An opposite explicit sector
  must change the physical subspace and its offset.
- Fail on malformed preparation inputs and inconsistent sector requests before
  optimization. Preserve rejection of trainable static-lifting inputs.
- Execute the documented CLI with optional quantum and chemistry packages
  blocked, and register its actual source in the existing example inventory.
- Run focused local scientific and documentation checks. Full Linux suites
  belong to PR CI. No accelerator, distributed, convergence-rate or scalability
  evidence follows from these bounded CPU tests.

## Ownership and remaining boundaries

The example owns only composition and its fixed illustrative model. Native
algebra/discovery, reduction, Clifford lowering, observable conversion, execution
and PyTorch retain their existing responsibilities and public entry points.
The selected sector is not inferred from particle number and need not contain
the global ground state. This is not an automatic full-register ansatz transform,
coefficient/SCF differentiation or a maturity promotion. Existing discovery's
catalog entry remains experimental with unsupported coefficient gradients.

The planned shortest contributor path is
`python -m examples.algorithms.fermion_pauli_training`, followed by
`python -m pytest tests/integration/test_fermion_pauli_training.py -q`.
These files will be added by the implementation commit; this scope record does
not claim they already exist.
