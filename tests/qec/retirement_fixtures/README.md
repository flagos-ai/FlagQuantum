# Pre-retirement evidence for surviving objects

These additional artifacts were exported with the pre-retirement **0.3.1**
installation, before installing the 0.4.0 implementation. They supplement the
original 15 artifacts in `../fixtures/`; none of those original bytes changed.
The source wheel was built from main commit
`4bf7a154efe4d40bb2216981efe503d1fb0e3793` (see the existing acceptance record), SHA-256
`e65c378c38786f3cace43e9a81abff0600a069b96d155e7425b134550fa0d941`.
The exporter ran outside the checkout, with `PYTHONPATH` unset, on Python 3.12
and verified every source fixture against the contract's frozen hashes first.

| Artifact | SHA-256 | Contents |
| --- | --- | --- |
| `surviving_instances.pkl` | `b92ca96f477120d3817002fd6c975aee0631b045fa03b8bc3fa3aa1b1b39f3f1` | Existing surviving objects from 003 (10), 005 (4), 006 (4), and the noise/sweep objects from 002 (2). Protocol 4; original types, attributes and tensor state. |
| `pre_retirement_matrices.json` | `3b88020b22e2dfaccae6de57e7040a049a68dada505157dcd2f36b047b7fec4c` | Original 002 `hz`, `hx`, `lz`, `lx` tensor dtypes and values, exported from the old pickle rather than recomputed by the new implementation. |

To reproduce the selection using a trusted pre-retirement installation, load
`restructure_003_codes.pkl`, `restructure_005_globals.pkl` and
`restructure_006_instances.pkl`. Select `payload.get("instances", payload)`
entries whose value type has a `flagquantum.qec.*` module outside the 23 retired
names. Store each selection under its original fixture filename. Add the old
002 `noise` and `sweep` objects under `restructure_002_matrices.pkl`, then dump
that outer dictionary with `pickle.dumps(..., protocol=4)`. Export the four
matrix fields as `{"dtype": str(tensor.dtype), "values": tensor.tolist()}`.
The contract JSON is the authoritative retired-name and original-hash list.
Tensor storage identifiers can make a newly exported binary hash differ; the
checked-in artifact above is the immutable positive compatibility evidence.

These are trusted repository test artifacts, never an unpickling bridge. The
original compound pickles now fail on removed lookup paths. Positive surviving
pickle tests and original numerical/state JSON comparisons continue separately;
new canonical checkpoints cover the removed-path definitions. Do not regenerate
these artifacts to accommodate an implementation change.
