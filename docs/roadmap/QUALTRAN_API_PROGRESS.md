# Qualtran API progress

The [inventory](QUALTRAN_API_INVENTORY.csv) tracks 232 entries. It contains
226 entries from the official [API Reference](https://qualtran.readthedocs.io/en/latest/reference/index.html)
and six explicitly tracked Bloqs from the algorithm roadmap.

The reference count is one row per documented symbol page under
`reference/qualtran/` in the [Sphinx inventory](https://qualtran.readthedocs.io/en/latest/objects.inv).
It excludes module pages and does not count class methods separately. The
supplement is an initial subset of the Bloqs library. This baseline is a
snapshot of the latest documentation retrieved on 2026-11-03, not a tagged
release. The inventory header does not declare a version.

Inventory SHA-256: `ce0637e400dd9281d4049524f2ac10c58cbda18b75ef364b222dca0d118632a1`

| Status | Count | Meaning |
| --- | ---: | --- |
| Unassessed | 226 | FlagQuantum mapping and evidence need review |
| Planned | 4 | Tracked in the algorithm roadmap |
| In review | 2 | Implementation has a PR |
| Verified | 0 | Merged and supported by parity or equivalent contract evidence |

Update the CSV with each PR. Record the FlagQuantum symbol and evidence link.
Mark a row verified after merge and validation against the agreed contract.
Record intentional semantic differences in the evidence field. Expand Bloq
coverage against a pinned upstream revision before reporting full-library parity.

## SelectedMajoranaFermion

[FlagQuantum #1035](https://github.com/flagos-ai/FlagQuantum/pull/1035) remains
in review. Three parity cases passed against [Qualtran 0.7.0](https://pypi.org/project/qualtran/0.7.0/)
with Cirq 1.4.0 and NumPy 1.26.4. Five local cases also passed.

The parity test uses one control, two index qubits and four targets. A seeded
complex input has amplitudes across both control branches and all four
addresses. It compares every complex output amplitude at an absolute tolerance
of 1e-12 and checks that Qualtran's decomposition returns the ancillary qubits
to zero.

FlagQuantum defines unused binary addresses as identity and verifies that
behavior locally. The upstream comparison covers the declared iteration domain.
Qualtran exposes typed selection registers and uses unary iteration.
FlagQuantum's reference path accepts flat binary bindings and materializes a matrix at
compile or execution time, capped at ten operand qubits. The stacked test
run passed all ten cases, including legacy IR and deferred lowering.

To run the optional test from a FlagQuantum development environment:

```sh
python -m pip install 'qualtran==0.7.0'
python -m pytest -q tests/unit/algorithms/primitives/multiplexers
```

## SelectPauliLCU

[FlagQuantum #1037](https://github.com/flagos-ai/FlagQuantum/pull/1037) passed
three complex-amplitude comparisons against Qualtran 0.7.0 on reordered
registers with no control, control on zero, and control on one. The cases use
a signed Pauli table. The local round-trip test also verifies identity on an
unused address. The current dense lowering supports ten operand qubits.
