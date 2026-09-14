# API Change Proposal 034: resumable Twin submissions

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner authorized the next Twin increment after convergence completion.

## Problem

Real QPU queues outlive Python processes. Before this proposal,
`TwinExperiment.submit()` returned the correct provider receipt, but callers had
no Twin-owned artifact that saved the frozen experiment and that receipt
together. Losing either object prevented a later process from validating the
original result without reconstructing identity-sensitive state by hand.

`fq.restore_job()` solves the general detached-job lifecycle, but its receipt
does not carry the frozen Twin prediction and experiment identity required for
Twin evidence. `TwinSubmission` composes that additional binding without
changing the Remote job contract.

## Public API

Complete first-process example:

```python
import flagquantum as fq
from flagquantum.remote.qpu import QuafuProvider

provider = QuafuProvider()
circuit = fq.Circuit(2).h(0).cx(0, 1)
chip_info = provider.fetch_chip_info("Shenglian")
twin = fq.twin.from_quafu_chip_info(
    chip_info,
    target="quafu:Shenglian",
    qubits=(20, 27),
)
experiment = fq.twin.TwinExperiment.prepare(
    twin,
    circuit,
    name="resumable-bell",
    shots=1024,
)

receipt = experiment.submit(provider)
submission = fq.twin.TwinSubmission.from_receipt(experiment, receipt)
fq.twin.dump_submission(submission, "twin-submission.json")
```

Complete later-process example:

```python
import flagquantum as fq
from flagquantum.remote.qpu import QuafuProvider

provider = QuafuProvider()
submission = fq.twin.load_submission("twin-submission.json")
status = provider.query_status(submission.receipt)
if status == "Finished":
    result = provider.fetch_result(submission.receipt)
    hardware_report = submission.validate_result(result)
```

`from_receipt()`, `dump_submission()`, and `load_submission()` are offline. They
never submit, poll, retry, cancel, or fetch a task. Provider calls remain explicit
in application code.

## Safety and identity

`flagquantum.twin_submission.v1` contains the complete frozen experiment and a
credential-free provider task receipt. Loading fails closed on unknown fields,
unsupported schemas, malformed nested objects, a different backend, a different
submitted-QASM digest, or a different ordered physical mapping.

The writer creates a private mode-0600 file, permits an idempotent write of the
same submission, and refuses to replace different or invalid content. The
loaded receipt payload is copied into immutable mappings and tuples so later
mutation of the caller-owned response cannot change the saved binding.

The artifact is not a credential store, scheduler, run manager, or proof that a
provider executed the requested program. `validate_result()` still applies the
existing retrospective, provider-attested checks before evidence can be made.

## Compatibility

This proposal adds `TwinSubmission`, `dump_submission()`, and
`load_submission()` without changing any frozen Twin v1 name, signature, field,
status, or serialization meaning. The new `flagquantum.twin_submission.v1`
schema is added to the machine-readable contract and protected by the same
compatibility policy.
