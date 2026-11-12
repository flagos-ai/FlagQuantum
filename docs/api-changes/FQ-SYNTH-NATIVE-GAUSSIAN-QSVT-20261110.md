# Native Gaussian QSVT LCU and exact amplification

Status: accepted locally. 18 new Gaussian tests and 21 DQI regressions pass;
29 catalog/API tests and strict typing of five Python files pass. Fresh 216
originals exactly match prior hashes, outcomes and resources; approved six pass
and all 29 production pins are unchanged. Native coverage is 201/216; Notebook
coverage remains 24/213.

Native: 18 physical bits, 5 compiler work bits, two explicit clean ports,
3,757 operations and depth 3,186. Approved source: 17 physical bits, 6 work
bits, 3,194 operations and depth 2,873. Independent full complex amplitude
errors are below 1e-14; clean-port and work leakage are below 1e-22.

Preserve eight fractional data bits, all 29 even and 28 odd LCU phases, the
source's preparation/adjoint order, branch-relative and absolute phases, and two
rounds of exact amplitude amplification. Expose the data as a raw eight-bit
array with the same fractional interpretation word/256; this changes its native
container, not its state or bit order. Do not substitute a Gaussian target state
for the quantum computation.

The frozen original is one of six retained semantic refusals: its declared
success amplitude 0.39579583583036 disagrees with the unchanged sequence's
observed 0.39191102421483637. Use only the already approved test-side revision
with the latter value for successful acceptance; retain the original refusal
and both source hashes. Reject the incorrect value in native construction too.
Do not silently fix it or credit unchanged-original execution.

Reuse native bound real arrays, mutable raw-register borrowing, CX/CRX,
rotations, phase polynomials, direct-X raw-array control and adjoint. A selected
inverse sine block may use an equivalent phase polynomial between X-axis basis
rotations; projector rotations combine even/odd angles in one polynomial to
avoid duplicate effects. Adjacent target Z-basis changes around commuting CRX
gates may be merged. No Compiler effect, helper, ownership or construction cap
is raised. No protected export or production behavior changes.

The complete native algorithm may expose clean attenuation and reflection
marker bits explicitly: prove both return to zero and report them separately
from compiler workspace. Avoid assuming a general native free/reset API.
An independent Circuit must use direct controlled-Y matrices and the retained
projector sequence, not the native phase polynomial; compare every complex
amplitude with the approved source and native circuit without ignoring global
phase. Verify original refusal, source/model identity, bit order, pre-amplified
marked overlap, coherent nonzero selected inverse, inverse restoration, clean
ports/work and budget refusal. Preserve original and approved resource baselines.

After final acceptance, run a fresh 216-original and approved-six replay,
archive immutable evidence and update the catalog and delivery ledger. Coverage
starts at 200/216; Notebook workflow coverage remains 24/213. Classification is
`single_device_fast_path`. Do not push, open an integration PR or run full remote
CI before all 216 native quantum-program acceptances are complete.
