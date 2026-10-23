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
- [surface/](surface/README.md): the registered rotated family, the independent
  historical frame and the checkerboard-conjugated ZXXZ family.
- `__init__.py`: historical exports, lazy `CssCode` alias and authoritative
  `_CODES` table. Keeping the registry here preserves existing plugin replacement
  and monkeypatch behavior; no forwarding registry or second table is introduced.

User entry points remain `flagquantum.qec` and their existing submodules.
Historical flat family modules explicitly re-export the same objects, including
previous helper aliases used internally. `_compat.py` retains historical
`__module__` values on moved definitions and their own methods/accessors. It
leaves Python/typing-owned protocol methods alone. No wrapper, alternate reducer,
safe-global allowlist or new class is created. `CssCodeMatrices` retains the 002
identity at `dem_matrices` and `dem_construction` too.

Core owns these compatibility paths. Removal has no approved target release;
it requires a separate approved deprecation and serialized-artifact migration
proposal. Physical source links follow the owning files; import and pickle
identifiers remain historical. Plugin code should continue importing public
entries; code and decoder registration behavior is unchanged.

For a check-layout change, edit its family module, run the shortest existing
user workflow, then the relevant family tests and compatibility tests:

```bash
PYTHONPATH=. python examples/qec/code_family_imports.py
python -m pytest tests/qec/test_restructure_compatibility.py tests/qec/test_restructure_boundaries.py tests/qec/test_restructure_code_families.py tests/qec/test_codes.py tests/qec/test_surface_code.py -q
```

The example prints equal check layouts, separate frame identity, the two old
pickle modules, Steane counts `7 6 2`, and zero logical error rate. The 001 and
002 fixtures remain frozen; the pre-003 fixture also protects class/function
pickles and method metadata. Never refresh fixtures to conceal drift.
