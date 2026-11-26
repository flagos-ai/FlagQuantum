# Many-body algorithms

Owns spin-chain Hamiltonian models, Heisenberg HVA and Trotter construction. General Hamiltonian construction and evaluation live in `../primitives/hamiltonian.py`. Do not duplicate operator algebra, simulation engines or device adapters. Published spin-model/HVA/Trotter entry points are unchanged.

## Ten-minute contributor path

Read the existing implementation and its scenario test, make one focused change,
then run from the repository root:

```sh
python -m pytest tests/unit/test_structure_import_compatibility.py -q
python tools/check_architecture.py
```

Keep existing public contracts and serialized identities. Physical accelerators
and scalability claims retain their separate acceptance gates.
