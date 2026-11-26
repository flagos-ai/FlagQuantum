# Quantum linear algebra

Owns the existing PCA and SVD algorithms. Shared numerical primitives remain in Simulation, and device adaptation stays in Compute. Published algorithm exports and result identities are unchanged.

## Ten-minute contributor path

Read the existing implementation and its scenario test, make one focused change,
then run from the repository root:

```sh
python -m pytest tests/unit/test_algorithms_pca.py -q
python tools/architecture/check_architecture.py
```

Keep existing public contracts and serialized identities. Physical accelerators
and scalability claims retain their separate acceptance gates.
