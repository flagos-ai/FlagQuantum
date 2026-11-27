# Runtime observability

Owns existing runtime observations, acceptance/evidence reporting and MPS critical-path profiling. Phase-aware stall diagnosis belongs here: progress snapshots, bounded compile/checkpoint budgets, and diagnosis of input, collective, memory and rank stalls. The watchdog uses the standard library and does not import Testing. Profile schemas and recorded values are unchanged. Do not manufacture timings or certify hardware from CPU fixtures. Depends on Core and existing Runtime evidence interfaces.

## Ten-minute contributor path

Read the existing implementation and its scenario test, make one focused change,
then run from the repository root:

```sh
python -m pytest tests/unit/test_mps_profiling.py tests/unit/test_correctness_properties.py -q
python tools/architecture/check_architecture.py
```

Keep existing public contracts and serialized identities. Physical accelerators
and scalability claims retain their separate acceptance gates.
