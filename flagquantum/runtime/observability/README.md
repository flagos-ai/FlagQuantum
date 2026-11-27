# Runtime observability

Owns existing runtime observations, acceptance/evidence reporting and MPS critical-path profiling. Profile schemas and recorded values are unchanged. Do not manufacture timings or certify hardware from CPU fixtures. Depends on Core and existing Runtime evidence interfaces.

## Ten-minute contributor path

Read the existing implementation and its scenario test, make one focused change,
then run from the repository root:

```sh
python -m pytest tests/unit/test_mps_profiling.py -q
python tools/architecture/check_architecture.py
```

Keep existing public contracts and serialized identities. Physical accelerators
and scalability claims retain their separate acceptance gates.
