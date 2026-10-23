# Surface code families

Core owns these layouts and their Pauli checks. Allowed dependencies are the
standard library, `codes.checks`, the narrow historical-metadata helper and QEC
Pauli algebra. Circuit generation, detector models, decoding and execution belong
to their own layers.

`rotated.py` defines the registered `flagquantum.qec.RotatedSurfaceCode`, whose
historical module is `flagquantum.qec.codes`. `_rotated_frame.py` defines the
independent class still imported from `flagquantum.qec.surface`. They have equal
check layouts and different class identities. `zxxz.py` owns checkerboard
conjugation and `ZxxzSurfaceCode`, retaining `flagquantum.qec.surface` identifiers.
The package facade exports the registered rotated family and the ZXXZ family;
the historical flat module continues exporting its independent frame class.

Keep both rotated definitions until a separately approved public-API and pickle
migration permits unification. Core owns the aliases; no removal release is
approved. The two lattice helpers remain beside their respective definitions to
preserve historical function identity and avoid changing numerical algorithms.

A typical layout change starts in the responsible family file. Run
`PYTHONPATH=. python examples/qec/code_family_imports.py`, then
`python -m pytest tests/qec/test_surface_code.py tests/qec/test_mixed_check_code.py tests/qec/test_restructure_code_families.py -q`.
Compare lattice order, CNOT order, logical representatives and both historical
pickle identities; agreement only on qubit counts is insufficient.
