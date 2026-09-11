# Phase 18 dynamic-noise bridge evidence

Date: 2026-09-09

## Accepted profile

The local dynamic executor accepts the existing backend-neutral `NoiseModel`
without creating a QEC-specific noise authority. Runtime owns random-stream
lifecycle, noise placement after matching executed gates, true-versus-observed
measurement flow, feedback from observed classical bits, and event accounting.
Simulation owns the numerical bit-flip and readout-sampling kernels.

The bounded profile supports independent one-wire bit-flip channels and
independent readout confusion. Physical state collapse uses the true
measurement result; the classical register and subsequent conditions use the
observed result. Readout rules also apply to final sampling. Both reference and
batched dynamic strategies implement these semantics.

The repetition-code adapter maps data bit flips after parity-check CNOTs and
readout confusion onto syndrome ancillas or final data wires. The middle data
wire participates in two check CNOTs per round, so it has two configured noise
opportunities; each edge wire has one. A finite-shot sweep records observed
logical outcomes and event counts without inferring suppression or a threshold.

## Evidence

| Gate | Result |
| --- | --- |
| Probability-one bit-flip noise reverses a matching X gate | pass |
| Physical collapse follows the true bit while observed feedback follows readout error | pass |
| Noise affects only shots on which a conditional gate executes | pass |
| Reference and batched strategies support the bounded noise profile | pass |
| Seeded batched noisy execution is reproducible | pass |
| Noise-model identity and channel/readout event counts are reported | pass |
| General Kraus channels fail before dynamic execution | pass |
| Correlated readout and device-profile timing noise fail closed | pass |
| Repetition-code noise placement reuses `NoiseModel` and existing channel types | pass |
| Stochastic QEC results retain syndrome, feedback, decoder, and logical records | pass |
| Finite-shot probability sweeps produce typed observation points | pass |

## Claim boundary

This evidence establishes one local stochastic dynamic-execution profile. It
does not establish arbitrary Kraus evolution, correlated readout, continuous
time, idle noise, calibrated hardware noise, leakage, crosstalk, noisy
gradients, real-time decoding, provider execution, statistical confidence
intervals, logical-error suppression, threshold behavior, scalability,
performance, or fault tolerance. Sweep points are observations for a specific
circuit-location model, not physical-device predictions.
