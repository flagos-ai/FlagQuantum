# Source workflow mapping and acceptance

Pinned source: `community/paper_implementation_project/bb84/bb84_implementation.ipynb`
at `b69c4893d8264d1e59ce0909df2bd88efed25e00`.
`provenance.json` records the source identity and all eight reviewed code cells.

| Source cells | Native workflow |
| --- | --- |
| 7, 8 | Frozen native qfunc; original eight signals and seed-3 message/Alice/Bob basis fixtures. |
| 9, 10 | Native synthesis and actual resource report; text circuit diagram. |
| 12, 13 | 2048 actual final observations from named `qba` readout; inspect most observed packet and observed rows. |
| 15, 16 | Explicit preparation inputs, basis reconciliation, original error threshold, matching-bit teaching filter and pairwise parity. |

The source final call omits its actual preparation arrays, causing regeneration
of unrelated message and bases. Explicitly pass the actual quantum inputs to
postprocessing. This is a documented correction, not identical reproduction of
the source's erroneous default call. Original seed-3 inputs are retained; local
sampling seed 3 is explicit and no identical remote sample sequence is claimed.
The source chooses a modal eight-signal packet; retain that selection. Native
count ties use ascending numeric word instead of a remote DataFrame's tie order.

The original source describes its matching-bit filter and pairwise XOR as error
correction/privacy amplification. Here they are clearly labeled illustrations;
they do not establish cryptographic security. Retain the strict error-rate
threshold 0.15. No matching bases yields error rate 1 and `None`; threshold
rejection yields `None`; fewer than two retained bits yields empty parity.

Numeric bit i represents signal i. Named readout and the workflow observation
record handle that convention; physical axes are reordered only in independent
acceptance tests. Zero temporary workspace does not mean the eight modeled
signals are zero. Classification: `single_device_fast_path`.

Acceptance covers three independent full absolute complex states and readouts,
every observed matched-basis bit, all eight signal mappings, actual 2048-shot
Notebook and CLI runs, reproducibility, no-match/threshold/short-key paths,
invalid inputs and resource refusals. Coverage is credited only after complete
local acceptance. No physical transmission, eavesdropper model or secure-key
certificate is implied by the simulated workflow.

## Recorded local acceptance outcome

44 focused pytest cases passed. A fresh local Jupyter kernel executed all four
clean native code cells plus one evidence-only assertion cell; the actual CLI
also sampled 2048 observations. The modal packet was numeric word 220, count 79.
Matching signal positions were (0, 2, 5), both sifted bit tuples were (0, 1, 0),
and error rate was zero. The reported pairwise parity (1,) is a teaching result,
not a certified secure key. Actual resources were eight modeled qubits, zero
temporary workspace, six operations and depth one after exact optimization.

All 216 original replay outcomes/resources and six approved revisions were
unchanged. The full default retained the prior 24 failed and 6 error nodes,
with 19619 passing tests; none were added or removed.
