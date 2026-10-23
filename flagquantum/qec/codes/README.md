# QEC code definitions

This package owns the existing stabilizer-code checks, protocols, factories and
single code registry. `matrices.py` owns CSS matrix records, binary validation and
code-to-matrix extraction. It must not import DEM construction, compiler,
runtime, simulation or experiment execution. Standard-library helpers, PyTorch
matrix operations, QEC algebra and code records are allowed dependencies.

The package `__init__.py` is the former `qec.codes` module: its public objects,
lazy `CssCode` export and mutable registry remain authoritative and unchanged.
Code-family splitting belongs to QEC-RESTRUCTURE-003. The independent class in
`qec.surface` remains distinct from `qec.codes.RotatedSurfaceCode`.

Existing public entries remain `flagquantum.qec` and `flagquantum.qec.codes`.
`CssCodeMatrices` and `css_code_matrices` are the same objects at the root,
`codes.matrices`, `dem_matrices` and `dem_construction`; their historical defining
module stays `flagquantum.qec.dem_matrices` for pickle compatibility, including
the relocated class's constructors, dataclass methods and property accessors. Core owns
these aliases; removal has no approved target release and requires a separate
approved deprecation proposal.

A short local workflow:

```python
from flagquantum.qec import CssCode, SteaneCode, css_code_matrices

matrices = css_code_matrices(SteaneCode())
code = CssCode(matrices=matrices, distance=3)
print(code.num_data_qubits, matrices.num_checks, matrices.num_observables)
# 7 6 2
```

For a matrix-validation change, edit `matrices.py`, retain its existing error and
normalization behavior, and run:

```bash
python -m pytest tests/qec/test_restructure_compatibility.py tests/qec/test_restructure_boundaries.py tests/qec/test_css_code.py tests/qec/test_dem_code_matrices.py -q
```

The unchanged 001 fixture protects public signatures, fields, identities and
historical pickle paths; never refresh it to conceal a migration failure.
