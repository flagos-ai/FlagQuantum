# The Twin region composer relabels through the one qubit-relabelling rule

**Date:** 2026-10-18
**Slice:** `N1-2`
**Surface:** the refusals a Twin region composition raises, and the private helper that
produced them
**Contract:** `contracts/circuit-composition-contract.toml`, section `[issue_codes]`;
`contracts/qubit-vocabulary-contract.toml`, section `[boundary]`
**Gate:** `tools/check_circuit_composition_contract.py`, function `_relabelling_errors`
**Conformance:** `tests/unit/test_twin_region_relabelling.py`
**Predecessor:** [Circuit composition](FQ-CIRCUIT-COMPOSITION-20261002.md)

## What this slice decides

`fq.twin.compose_region_twin` places several local Twin cells onto one connected
region. Every cell records its calibration, its gate durations, its gate-noise rules and
its readout rules in the cell's **local** qubit labels, so composing a region means
relabelling all four of them onto the region's labels.

That relabelling now has one implementation, in Core, and the composer reads it:
`flagquantum.core.qubit_mapping.remap_qubits`. The private helper the composer used to
carry — `flagquantum.twin.region_model._remap_wires` — is deleted, together with the
second, hand-written bounds check that produced the second of the two refusals below.

The composition contract records which package modules read that rule, and the gate
measures the record against the import graph on every run. A module that copies the rule
without importing it, and a recorded reader that stopped reading it, both fail.

## Why the copy was worth deleting

The two implementations were not different. `_remap_wires` was

```python
def _remap_wires(wires, *, local_to_region):
    try:
        return tuple(local_to_region[wire] for wire in wires)
    except KeyError as error:
        raise ValueError("Twin noise rule references a wire outside its mapping") from error
```

and `remap_qubits` is the same lookup with the refusal sentence assembled from an
`owner` argument. A second implementation of the *same* rule is more dangerous than a
second implementation of a *different* one, because there is nothing to distinguish them
by: they agree until one of them is edited, and then the divergence is invisible until a
user reports a message that no longer matches the code.

The copy also mis-attributed two of its four callers. `_remap_wires` had exactly three
call sites, and its one message said `Twin noise rule` at all of them:

| Call site | What it relabels | Message before | Message now |
| --- | --- | --- | --- |
| `_compose_device_profile`, gate durations | a gate duration's qubit scope | `Twin noise rule …` | `Twin device profile …` |
| `_compose_noise_model`, gate-noise rules | a rule's qubit scope | `Twin noise rule …` | `Twin noise rule …` |
| `_compose_noise_model`, readout rules | a readout rule's qubit scope | `Twin noise rule …` | `Twin readout rule …` |

A user who scoped a gate duration — or a readout rule — to the wrong qubit was told that
a *noise rule* was wrong. The sentence was the helper's, not the caller's, and a helper
that serves three callers cannot name any of them correctly.

## What a user sees now

The refusal text is documented behaviour, so it changed on purpose and it changed
together with its vocabulary. Both sentences are asserted exactly, not by substring,
in `tests/unit/test_twin_region_relabelling.py`; a message that calls a qubit a "wire"
again is a vocabulary regression, and exact equality is what makes the regression
visible.

```text
before  Twin noise rule references a wire outside its mapping
after   Twin noise rule references qubit 9, which the qubit map does not name

before  Twin device profile references a wire outside its mapping
after   Twin device profile references qubit 9, which the qubit map does not name
```

The second sentence also says *which* qubit, which the first one did not: the old bounds
check compared a label against the mapping and then the helper looked the label up, so
the label was known and not printed. The new sentence is assembled by `remap_qubits`
from the same three facts at the same place, and it names the qubit, the subject and the
map that does not name it.

Only `ValueError` is raised, at every site, before and after. No exception class moved.

## The rule the contract now records

`contracts/circuit-composition-contract.toml` already named `remap_qubits` as an
internal-only refusal source, because `Circuit.compose` always passes the dense tuple
that `qubit_map_from` produced and therefore never reaches the lookup refusal. This
slice adds the missing half of that statement: *who reads the rule*.

```toml
consumers = ["flagquantum/circuit.py", "flagquantum/twin/region_model.py"]
retired_wording = "outside its mapping"
```

The gate does four things with those two keys, and each one is negative-tested in
`tests/unit/test_circuit_composition_contract.py` against a package it builds itself:

* every recorded reader exists and calls the owner by name, so a module cannot be
  recorded as a reader it is not;
* the recorded reader list equals the modules under `flagquantum/` that import the owner
  from the rule's module, resolved through relative imports, so a *new* reader cannot
  appear unrecorded and a *lost* reader cannot stay recorded;
* the retired private wording does not occur anywhere under `flagquantum/`, so a reader
  that folds a failed lookup into its own refusal is caught even if it never mentions the
  rule again — which is exactly the shape that was deleted here;
* a row that records no readers at all is refused, so the clause cannot be satisfied by
  being empty.

The scan is the shipped package, not the repository: a test may call the rule directly
without becoming an implementation of it. The one remaining occurrence of the retired
wording is inside the predecessor record's code quotation, which is a historical
statement about the helper and is not rewritten.

## The private bucket moved by one

`flagquantum/twin/region_model.py::_remap_wires::wires` was the only private wire-named
parameter this slice removed, and `[boundary] measured_private` fell from 226 to 225 as a
consequence. That number is a census, not a threshold: the public buckets did not move
(`measured_canonical` 341 and `measured_deprecated_aliases` 11 are unchanged), and the
gate compares all three for equality, so a change in the public surface could not be
absorbed by this one.

```console
$ python -c "from pathlib import Path; from tools.census_wire_vocabulary import census; \
             import flagquantum; print(len(census(Path(flagquantum.__file__).resolve().parent).internal))"
225
```

At the branch point `6a6c080be` the same command prints `226`, and the removed entry is
`flagquantum/twin/region_model.py::_remap_wires::wires`.

## Evidence

```console
$ python -m pytest tests/unit/test_twin_region_relabelling.py -q
7 passed

$ python tools/check_circuit_composition_contract.py
Circuit composition contract passed

$ python tools/check_qubit_vocabulary.py
Qubit vocabulary contract passed: 341 of 341 baseline sites retired, 11 kept as deprecated aliases
…
Qubit vocabulary call sites: 0 keywords passed that their callee rejects (invariant: 0)

$ python -m pytest tests/unit/test_census_wire_vocabulary.py \
      tests/unit/test_qubit_vocabulary_contract.py -q
172 passed
```

The refusals were measured against the public entry point rather than argued. The
drivers live in `tests/unit/test_twin_region_relabelling.py`; the shortest probe that
reaches a refusal from `fq.twin.compose_region_twin` is

```console
$ python -m pytest tests/unit/test_twin_region_relabelling.py -q -k refused
4 passed, 3 deselected
```

Four of those seven name a refusal, one per scoped object — a gate duration, a gate-noise
rule, a readout rule and a calibration — and each asserts its sentence by exact equality:

```text
Twin device profile references qubit 9, which the qubit map does not name   (duration, calibration)
Twin noise rule references qubit 9, which the qubit map does not name       (gate-noise rule)
Twin readout rule references qubit 9, which the qubit map does not name     (readout rule)
```

The remaining three pin the same relabelling when it succeeds: a region of physical
qubits `(12, 13)` composed from a cell whose local qubits are `(0, 1)` reports
durations scoped to `(0, 1)`, an unscoped duration stays unscoped, and a gate-noise rule
and a readout rule move with the same rule as the durations do.

## What this slice does not reach

* The composed objects themselves — the merged calibration, the merged durations, the
  merged noise and readout rules, and the region's coverage check — are measured by
  `tests/test_twin_region_model.py`, which is an `integration` file over the same entry
  point. This slice adds a `unit` file beside it for the relabelling rule and its
  refusals, and does not restate what that file already owns.
* The four other places a qubit sequence is relabelled inside the package —
  `runtime/executors/statevector/program_cache.py`, `simulation/tensor_network/path_search.py`,
  `simulation/statevector/product_state.py` and `compiler/routing.py` — are *different*
  operations, not copies of this one: they rebuild an instruction with routing metadata,
  search a tensor-network path, relabel swap components, or re-emit a routed instruction.
  They are not folded in here, and the readers ledger is scoped to `remap_qubits` so it
  does not claim they are.
* The composer still reads a legacy noise object's own `wire` and `wires` fields. Those
  spellings are accepted input from third-party and recorded artifacts, in the same class
  as the drawer's accepted `n_wires`, and they are not renamed here.
