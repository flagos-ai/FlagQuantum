# Portfolio inequality workshop acceptance

Retain 4 layers / scale 1.4 / penalty 1.3 / 20 calls and 3 layers / scale 10 / penalty 30.5 / 20 calls. All assets remain unsigned 2-bit values and the numeric slack is 3 bits (0..7), with weights 1/2/4. There are 512 joint assignments. Cost must include slack before any marginalization; sum(assets)<=6 and sum(assets)+slack==6 are distinct diagnostics.

Preserve every original scenario and problem size. Users supply high-level
costs and classical parameters; the native qfunc prepares uniform registers,
applies positive cost phases, and mixes with RX(beta). No physical wires or bit
decoding are required in the workflow. The companion portfolio_programs.py shows
the complete native quantum functions.

The source uses SciPy COBYLA, whose maxiter limits objective evaluations. Native
COBYLA counts trust-region iterations, so this workflow stops BEFORE an extra
objective execution at the source cap, selects the best actually evaluated
parameters, and reports evaluation_budget and nonconvergence. It does not invent
an optimizer return value or claim the same optimization trajectory. Exact local
mean probabilities replace provider shot-noisy estimates; final 1,000 seeded
samples are actual draws. Sampling/readout and independent verification are
outside the objective-call cap. Classical exhaustive minima are comparison labels,
never substituted for quantum observations. Penalties stay soft; results retain
infeasible assignments. No remote service, hardware or advantage claim.

Each scenario compares its original initial schedule, two nonzero signed
parameter bindings and its actually selected parameters against independent
Circuit factor phases and RX mixers. Every absolute amplitude, bit order, global
phase and clean workspace is checked. Enumerate all 64 or 512 joint costs against
an independently expanded covariance expression. Compare the final scenario with
the unchanged native quantum parent. Check operation/memory/width refusals and
strict objective-call bounds. Fresh-kernel execution covers every mapped source
code cell and inspects all trace and sampled-cost figures. All 216 original
programs and six approved alternatives are replayed before the frozen default
test tier. Coverage credit follows terminal acceptance, never preflight.
