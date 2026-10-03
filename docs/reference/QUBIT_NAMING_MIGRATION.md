# Qubit keyword migration

Implementation candidate on `qubit-naming-migration`; pending API/release review.

New code uses `fq.Circuit(n_qubits=2)`, `fq.probabilities(qubits=(0,))`,
`fq.samples(qubits=(0,))`, `fq.counts(qubits=(0,))` and
`fq.RuntimePolicy(observable_qubits=(0,))`. Cloud deployment profiles use
`CloudBackendProfile(n_qubits=...)` and expose `profile.n_qubits`. Remote
providers list hardware and simulators with `list_devices(n_qubits=...)`.

Old count aliases and `wires` / `observable_wires` keywords emit
DeprecationWarning. Supplying both selection keywords is an error, including
explicit None. Positional selection and the Observable overload are unchanged.
The proposed schedule is deprecation in 0.3.x and removal in 0.4.0, subject to
release-owner review; this PR does not bump the package version.

RuntimePolicy writes schema `flagquantum.runtime_policy`, version `2.0`, with
`observable_qubits`. Its reader accepts the prior unversioned payload with
`observable_wires`, including module state dictionaries. Conflicting selection
fields and unsupported versions are rejected. Older package versions do not
understand the new writer; checkpoint compatibility is backward-reading, not
forward-reading.

IR and backend-native payload fields are unchanged. The deprecated
`observable_wires` property remains available on RuntimePolicy, while
`CloudBackendProfile.n_wires` and its constructor keyword remain compatibility
aliases during the same migration window. `discover_backends(n_wires=...)`
likewise delegates to `list_devices(n_qubits=...)` with a deprecation warning.
The affected candidate signatures
are updated for this explicitly requested migration; the historical baseline
and checker remain unchanged.

## Scope of the rename

The migration covers four surfaces, all measured by
`tools/census_wire_vocabulary.py` and reconciled against
`contracts/qubit-vocabulary-contract.toml` on every run:

| Surface | Count | Disposition |
|---|---:|---|
| parameters on the public function surface | 341 | renamed; 11 are already deprecated aliases and are deleted at 0.4.0 |
| public attribute and property names | 144 | 122 renamed, 22 excluded as payload keys |
| module-level public definition names | 10 | renamed |
| string literals | 1291 | reported only, never ledgered |

An attribute is renamed whether it is a field, a property or method
(`Circuit.n_wires`), or an instance attribute assigned in a method
(`TextDrawer().wire_order`). It is excluded only when its spelling reaches a
payload, and each exclusion names the class or method that witnesses it plus the
schema event that retires it — `IR_VERSION` for the IR keys, or the owning payload
schema's own version bump. See
[the attributes authorization](../api-changes/FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md)
and [its decision record](../development/API_CHANGE_PROPOSAL_067_QUBIT_VOCABULARY_ATTRIBUTES.md).

Aliases are owed only where the name is reachable from `fq.*`. On the attribute
surface that is `fq.Circuit.n_wires`, `fq.MeasurementResult.wires`, and
`fq.OutputRequest.wires`; everything else is renamed in place in the same release.

## Spellings the census cannot see

The scanner reads names — parameters, attributes, definitions — and reports
string literals without judging them. Two further user-visible spellings are
reachable exactly the way a parameter is, and neither is on a ledger. Both are
listed here so that "the ledger is clean" is not read as "no user-visible `wire`
is left":

| Spelling | Where a user meets it | Disposition |
|---|---|---|
| `wire_options`, `show_wire_labels`, `active_wire_notches`, `n_wires` | keyword arguments to `Circuit.draw(**kwargs)` and `draw_mpl(**kwargs)`, which forward to the drawers instead of declaring a parameter | open; the drawer docstrings document the accepted spelling, which is still the old one |
| `wires` | the keyword a captured hybrid program must use — `qp.H(wires=...)`, `qp.measure(wires=...)`, `qp.reset(wires=...)` — required by the capture layer, which rejects any other keyword | open; renaming it changes the source language, not a signature |

Neither is renamed here. Renaming the first would make the drawer docstrings
describe keywords the code does not accept; renaming the second would break every
hybrid program the capture layer can read. Both are candidates for their own
change, and the capture keyword is a decision for the hybrid-language owner
rather than for this migration.
