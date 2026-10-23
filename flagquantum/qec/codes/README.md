# QEC code definitions

Core owns code records, check layouts, matrix conversion, family construction and
the existing single code registry. These modules may use standard-library helpers,
PyTorch matrix operations and QEC algebra (`gf2`, `pauli`, `logical`). They must
not depend on DEM construction, compiler, runtime, simulation or experiment
execution. The QEC root facade still eagerly imports execution; this boundary
does not make a cold root import infrastructure-free.

- `checks.py`: check record, stabilizer protocol and ancilla bands.
- `repetition.py`, `steane.py`: fixed stabilizer families.
- `css.py`, `_css_helpers.py`, `matrices.py`: matrix records and CSS validation.
- `colour.py`, `toric.py`, `qldpc.py`, `bicycle.py`, `reed_muller.py`: factories.
- `subsystem.py`, `floquet.py`: gauge and measurement-period records.
- [surface/](surface/README.md): the single registered rotated family and the
  checkerboard-conjugated ZXXZ family.
- `__init__.py`: historical exports, lazy `CssCode` alias and authoritative
  `_CODES` table. Keeping the registry here preserves existing plugin replacement
  and monkeypatch behavior; no forwarding registry or second table is introduced.

User entry points remain `flagquantum.qec` and the responsibility packages.
The approved 0.4.0 retirement removes the flat family facades and their pickle
lookups. `CssCodeMatrices` now uses `codes.matrices`; family definitions use
their canonical owners. `_compat.py` retains metadata only for surviving
`qec.codes`, `qec.dem`, `qec.decoders`, `qec.adapters` and `qec.noise` identifiers.
It leaves Python/typing-owned protocol methods alone. No replacement reducer,
allowlist or registry is introduced. See the
[retirement contract](../../../docs/development/QEC_RETIREMENT_CONTRACT.md).

For a check-layout change, edit its family module, run the shortest existing
user workflow, then the relevant family tests and compatibility tests:

```bash
PYTHONPATH=. python examples/qec/code_family_imports.py
python -m pytest tests/qec/test_restructure_compatibility.py tests/qec/test_restructure_boundaries.py tests/qec/test_restructure_code_families.py tests/qec/test_codes.py tests/qec/test_surface_code.py -q
```

The example prints one class identity, equal check layouts, the surviving
`flagquantum.qec.codes` pickle module, Steane counts `7 6 2`, and zero logical error rate. The 001 and
002 fixtures remain frozen; the pre-003 fixture also protects class/function
pickles and method metadata. Never refresh fixtures to conceal drift.
