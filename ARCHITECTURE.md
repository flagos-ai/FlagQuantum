# FlagQuantum source architecture

```text
flagquantum/
├── api.py                 # stable user-facing API
├── core/                   # IR, gates, devices, compilation
├── runtime/
│   ├── backends/           # statevector, MPS, tensor-network, and JAX
│   ├── distributed/        # topology, protocols, adapters, and execution
│   ├── audit/              # schema, validation, statistics, and release
│   ├── observability/      # evidence and performance records
│   └── *.py                # execution, training, planning, and configuration
└── testing/                # reusable correctness/test helpers

benchmarks/
├── runners/                # reproducible execution and JSON contracts
├── research/               # exploratory analysis and plotting
└── legacy top-level scripts# compatibility entry points during migration
```

New user code should use `import flagquantum as fq` or the named runtime
subpackages. New benchmark code should use `benchmarks.runners`; research
scripts must not become runtime dependencies.
