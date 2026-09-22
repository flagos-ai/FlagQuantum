# Amazon Braket circuit interoperability contract

FlagQuantum will support `braket.circuits.Circuit` through an optional adapter
owned by `flagquantum.ecosystem.braket`. Braket circuit objects stop at that
package boundary and are converted immediately to FlagQuantum IR. The existing
`flagquantum.remote.qpu.braket` module remains the separate owner of device
discovery, task submission, polling, and result normalization.

The implemented adapter exposes static circuit conversion at this boundary. It
does not submit an AWS task, access credentials, select a device, or claim
compatibility with a real simulator or QPU.

## Dependency decision

The adapter will reuse the existing `braket` extra and its
`amazon-braket-sdk>=1.117,<2` dependency. Amazon Braket SDK 1.117.0 and the
current 1.127.1 release both require Python 3.11 or newer, so the requirement is
guarded by the same Python marker. Core FlagQuantum stays usable on Python 3.10
without installing Braket.

Both boundary versions run in an isolated SDK matrix. The SDK is Apache-2.0
licensed, remains optional, and can be removed with the adapter without
changing FlagQuantum IR or stable core APIs.

## Semantic boundary

Version 1 covers bidirectional conversion between immutable FlagQuantum IR and
static Braket circuits. Integer Braket qubit indices map directly to
FlagQuantum wires. Conversion preserves instruction order, bound finite real
parameters, and explicit wire extent. Statevector conformance must use ascending
Braket qubits and verify that wire zero is the most-significant axis.

Measurements and result types belong to an execution plan. AWS task submission,
device selection, rewiring, calibration, pulse gates, compiler directives,
verbatim boxes, noise, free parameters, and global phase remain outside this
adapter. Unsupported input fails closed unless a future implementation exposes
an explicit lossy conversion and records every loss in its report.

## Verification

The adapter must continue to:

1. convert the declared static gate subset in both directions;
2. reject or explicitly report every unsupported feature;
3. preserve otherwise idle wire extent without changing circuit semantics;
4. prove deterministic complex128 statevector equivalence and round trips;
5. test Amazon Braket SDK 1.117.0 and 1.127.1 in isolated Python 3.11+ lanes;
6. prove importing core FlagQuantum does not import Braket; and
7. keep cloud submission and credentials inside the existing Remote provider.
