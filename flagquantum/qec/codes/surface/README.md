# Surface code families

Core owns these layouts and Pauli checks. Allowed dependencies are the standard
library, `codes.checks`, retained code metadata and QEC Pauli algebra. Circuit
generation, detector models, decoding and execution belong to their own layers.

`rotated.py` defines the only `RotatedSurfaceCode`: the original registered
`flagquantum.qec.RotatedSurfaceCode`. Root, `qec.codes`, `codes.surface`, the owner
module and `get_code("rotated_surface", ...)` select this same class. Its pickle
module remains `flagquantum.qec.codes`. `zxxz.py` owns checkerboard conjugation
and `ZxxzSurfaceCode`; its canonical pickle path is now in this module.

The approved 0.4.0 retirement removes `qec.surface` and `_rotated_frame.py`.
Old surface imports and their pickle lookups fail; no `rotated_frame` public
replacement, subclass, alias module or unpickling bridge is added. ZXXZ uses the
existing rotated lattice helpers. Checks, lattice order and logicals stay equal
to the frozen numerical evidence.

For a typical change, edit the family owner, run
`PYTHONPATH=. python examples/qec/code_family_imports.py`, then
`python -m pytest tests/qec/test_surface_code.py tests/qec/test_mixed_check_code.py tests/qec/test_restructure_code_families.py -q`.
Compare lattice order, CNOT order, logical representatives, canonical checkpoint
round-trips and surviving registered-class identifiers. Keep original fixtures.
