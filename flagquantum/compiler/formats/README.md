# Code generation and interchange

Compiler owns this directory. It translates circuit instructions to OpenQASM,
QCIS and QIR text and reads supported OpenQASM back into native circuits.
`translate.py` dispatches formats, individual emitters own syntax, and
`target_emission.py` emits an already legalized target program. Target legality
and deployment packaging belong to their respective Compiler and Deployment
owners.

Allowed dependencies include Core IR and operator semantics, Compiler lowering
and target records, and the native Circuit type for import. These modules must
not choose execution backends, execute circuits or submit remote jobs.

The expert entries remain `flagquantum.compiler.translate`,
`flagquantum.compiler.openqasm.emit_openqasm` and the existing `openqasm_import`,
`qcis` and `qir` module paths under `flagquantum.compiler`. Change syntax in its
owning emitter or importer and add a round-trip or rejection case in the
matching test file.

```bash
python -m examples.synthesis_offline_deployment
python -m pytest -q tests/test_openqasm_compiler.py tests/test_openqasm_import.py tests/hybrid_compiler/test_translate.py tests/hybrid_compiler/test_qir_emission.py tests/integration/test_synthesis_offline_deployment.py
```

The example captures a Bell-state Python `@qfunc`, emits OpenQASM 3, and prepares
an offline DeploymentPackage. The integration test verifies the emitted state
after reading the text back. No credentials or hardware are required.
