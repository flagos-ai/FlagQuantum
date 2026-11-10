# Computational-basis measurement copy leaf

Simulation owns `tensor_network/measurement_copy.py`. Its
`z_measurement_copy_node(input_label, result_label, environment_label, like)`
ports follow unchanged Qualtran v0.7.0 `MeasureZ.my_tensors`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`: input quantum bit, classical result,
then hidden measurement environment. Only entries 000 and 111 are one.
Copyright and source attribution remain in the owner.

The caller supplies three distinct nonnegative integer labels and owns their
allocation and graph connectivity. Invalid labels fail before data construction.
Dtype/device follow `like`. The helper returns the existing TensorNetworkNode;
it creates no contract, registry, dispatcher, foreign sentinel or public export.
The source random environment label is replaced by the caller's explicit edge.
Native label validation is a deliberate earlier refusal, not identical source
failure timing for malformed connections.

The raw leaf is a coherent copy. To obtain the measurement channel, double it
with `tensor_conjugate_pairs`, then call `trace_conjugate_indices` on the original
environment label. Keeping the result retains the measurement record; tracing
its label additionally discards that record. Existing
`contract_register_superoperator` groups output-forward/output-backward and
input-forward/input-backward axes. No normalization, sampling, physical reset,
workspace allocation or graph traversal is performed by this numerical leaf.
Mixed classical/quantum component graph supply remains open.

The two previous hand-written copy tensors in the trace and superoperator
composition tests now consume this owner. Independent original leaf data can
replace it without changing the channel consumer. All four floating/complex
dtypes match original axis order, data and full measurement/discard channels.
Complex amplitudes of both outcomes, spectator coherence and an input phase
derivative are tested. A plus-state example proves that retaining the environment
is a pure copy, whereas tracing it produces rank-two classical output. This
prevents accidental use of the raw tensor as a finished measurement.

Physical users retain `circuit.append(op)`. Existing owned measurement tests
still compare all four unnormalized branches of the two-measurement circuit
against the original instrument, including reordered classical outputs and zero
work amplitudes. That circuit emits two measurements, no reset and no T gate.
The numerical supplier emits zero physical instructions/work. One source
MeasureZ model estimates one measurement, zero T-equivalent cost and peak one
bit; the two-measurement caller uses four physical bits with one clean work bit.
Numerical copy storage is 32/64/64/128 bytes for
float32/float64/complex64/complex128, with one explicit environment edge.

A short numerical construction is:

```python
import torch
from flagquantum.simulation.tensor_network.measurement_copy import z_measurement_copy_node
from flagquantum.simulation.tensor_network.conjugate_pairs import tensor_conjugate_pairs
from flagquantum.simulation.tensor_network.trace_indices import trace_conjugate_indices

leaf = z_measurement_copy_node(3, 17, 41, torch.empty((), dtype=torch.complex128))
channel_nodes = trace_conjugate_indices(tensor_conjugate_pairs((leaf,)), (41,))
assert len(channel_nodes) == 2
```

GRAPH72 verification: 19 new cases passed in 1.38 s; 1230 related cases in
24.19 s (41 existing/source warnings), 507 focused coverage cases in 28.77 s
(39 warnings), and 64 integration cases in 5.51 s. The new owner has 11
statements / four branches at 100%. Fourteen related owners total 405 statements
/ 170 branches at 99%, retaining two historical GRAPH60 refusals. Strict typing
passed across 31 related files and three source-following files; Black/Ruff,
scope, architecture, Program IR and public API checks passed. Twenty source
blobs match the locked inventory; 23 unchanged original tests passed in 1.14 s
(five existing warnings). Qualtran 0.7.0, Kickmix 0.1 Simulator and the
binary-field example passed. Evidence is under
`ftqc-reference-env/g2-measurement-copy-*`.

Default smoke/unit: 6 failed, 21485 passed, 300 skipped, 1037 warnings in 167.60s (0:02:47). All 6 observed failure IDs belong
to the independent seven-failure baseline; 1 were not reached before
maxfail. The default suite is not certified as fully passing.

## X-basis instrument (GRAPH73)

`x_measurement_node` takes the same explicit input/result/environment labels and
prototype. Its returned axes are **input/environment/result**, matching unchanged
`MeasureX.my_tensors` rather than the Z leaf's input/result/environment order.
The original exactly representable +/-0.5 entries are retained directly and
compared by array equality in all four dtypes. Both helpers share label validation;
no extra copy tensor is allocated just to validate the X labels.

The source supplies this tensor directly. It does not declare an H->MeasureZ
primitive decomposition. H->MeasureZ is the existing FlagQuantum physical
implementation. Independently contracting H -> shared Z copy -> H on the hidden
environment reproduces the source raw instrument within floating-point tolerance.
Dropping that environmental H changes raw environment amplitudes but leaves the
traced channel unchanged. That environmental basis change is numerical evidence,
not an additional physical gate. The direct source entries avoid factorization
roundoff in the returned leaf itself.

Nine new cases compare four dtypes with retained/discarded classical results,
source axis data, complete source channels and independent supplier replacement.
The unfused three-node factorization and the unrotated two-node representation
feed the same existing doubled-label/trace consumer. Every complex outcome
amplitude and relative-input-phase derivative is compared with the original
instrument. Existing `circuit.append(op)` emits the bound H+measure instructions
unchanged. Forced branches retain a Z eigenstate on the consumed physical wire;
rotating that wire only for amplitude comparison restores the source environment
basis. Both outcomes preserve the spectator and leave the clean work wire zero.

Resource reporting keeps distinct meanings: the returned X leaf has eight
entries, occupying 32/64/64/128 bytes in float32/float64/complex64/complex128.
These are returned storage sizes, not measured construction peaks. Numerical
supply adds zero physical gates/work. Actual lowering emits H1 + measurement1,
reset0 and T0. The unchanged source model reports measurement1, Clifford0,
T-equivalent0 and peak1. There is no source primitive recipe to claim equal to
this physical lowering. The caller uses three physical bits including spectator
and one clean bit. Earlier descriptions of an upstream H->MeasureZ decomposition
are superseded by this audited distinction.

GRAPH73 final verification: 1241 related cases passed in 14.67 s (41 existing/
source warnings), 516 focused cases in 18.45 s (39 warnings), and 64 integration
cases in 3.98 s. The measurement owner has 18 statements / four branches at
100%; fourteen owners total 412 statements / 170 branches at 99%, retaining the
two historical GRAPH60 refusals. Strict typing passed for 32 related files plus
three source-following files; Black/Ruff, scope, architecture, Program IR and
public API checks passed. Twenty-two original blobs match the inventory;
24 unchanged original tests passed in 1.26 s (five existing warnings), including
the original X measurement supertensor test. Qualtran 0.7.0, Kickmix 0.1 Simulator
and the binary-field example were exercised. Evidence is under
`ftqc-reference-env/g2-x-measurement-tensor-*`.

GRAPH73 final default smoke/unit: 7 failed, 21507 passed, 300 skipped, 1037 warnings in 120.19s (0:02:00). All 7 observed failure
IDs belong to the independent seven-failure baseline; 0 were not reached
before maxfail. This is not a fully passing default suite.
