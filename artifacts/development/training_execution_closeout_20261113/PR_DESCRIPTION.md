# Bind execution inputs and preserve parameter training state

Ordinary static circuits can accept an entry state or named scalar angle values
per execution and replay a compatible accepted plan. Qualified checkpoints now
preserve frozen parameter flags and ordered optimizer bindings, so resuming
training keeps the intended trainable parameters and optimizer moments.

The review series records the approved contracts and independently executable
replacement fakes before Runtime implementation. Separate commits repair native
CPU rotation/CX fusion capability admission and make missing-revision tests
independent of the producing checkout's private refs. Historical API snapshots
and evidence origin declarations remain unchanged.

## User example

```bash
PYTHONPATH=. python -m examples.local.train_parameters
```

Expected output:

```text
Frozen encoder unchanged; readout trained; checkpoint resumed; encoder unfrozen.
```

## Support boundary

Execution inputs cover exact, unbatched CPU PyTorch statevector execution from
ordinary static circuits. External states and named angle inputs are separate
paths in this slice. Plans retain input requirements rather than supplied
values or live autograd graphs. Checkpoint v3 covers qualified local workflows;
its compatibility and transactional failure rules are specified by the approved
contract. Capability maturity remains experimental.

## Validation

The local report and exact retained results are in
`artifacts/development/training_execution_closeout_20261113/validation.json`. Contract-only
replacement tests passed before implementation, and the user workflow, API
baseline, capability checks and focused regression tests have been exercised.
Network-dependent tests remain blocked in this sandbox and require a capable
runner before merge. No full-tier or distributed scalability certification is
claimed.

Fresh results: 499 focused passes (9 optional skips), 415 real-native passes,
and 92 consumer/compatibility passes. The full default-tier report before the
test correction retains 19,184 passes, 778 skips, 10 failures and 6 errors.
Four old writer-version assertions pass in the focused recheck; twelve cases
remain blocked by sandbox networking. The origin gate passes in the isolated
main-based checkout with unchanged validators and declarations.

Final packet privacy, origin and provenance checks: 95 passed.
