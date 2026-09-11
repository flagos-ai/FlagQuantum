# Phase 19 Runtime decoder-feedback evidence

Date: 2026-09-09

## Accepted profile

The local dynamic trajectory executor accepts a private bounded feedback plan.
Each decision point names already-recorded classical bits and invokes a
stateless controller with the complete observation history for that shot.
Runtime records physical measurement values separately from values changed by
readout confusion. The controller may return no action, a physical X, or an X
Pauli-frame update within the plan's declared wires and modes.

The repetition-code adapter translates observations into ordered syndrome
rounds and calls a replaceable `StreamingDecoder` after every round. Physical
mode changes the continued quantum state. Frame mode leaves the state unchanged,
uses the pending frame to interpret later parity checks, and applies the final
frame to data readout. The older compiled lookup and offline analysis modes
remain separate comparison paths.

## Evidence

| Gate | Result |
| --- | --- |
| Physical Runtime feedback changes the continued state | pass |
| Runtime frame feedback changes interpretation without changing state | pass |
| True bits, observed bits, action, and frame before/after are shot-resolved | pass |
| A replacement streaming decoder changes the executed correction and outcome | pass |
| Single errors on every data wire and verified round are corrected in both Runtime modes | pass |
| Compiled lookup, Runtime physical feedback, and Runtime frame feedback agree on the bounded logical outcome | pass |
| `auto` selects trajectories when decoder feedback is present | pass |
| Explicit batched feedback fails before execution | pass |
| Missing measurement points and actions outside the plan fail closed | pass |

## Claim boundary

This is a synchronous local statevector reference loop for one fixed
three-data-qubit repetition-code profile. The feedback records are private
Runtime subinterfaces, not stable public plugin types. The evidence does not
establish asynchronous decoding, controller latency, deadlines, provider or
hardware feedback, general stabilizer codes, measurement-error-tolerant temporal
decoding, noisy gradients, logical-error suppression, threshold behavior,
scalability, performance, or fault tolerance.
