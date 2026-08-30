# Payload naming convention

New artifacts should use:

```text
<backend>_<topology>_<qubits>q_<layers>L_<world>gpu_<mode>_<revision>.json
```

Examples:

```text
flagquantum_ring_31q_8L_16gpu_optimized_rerun1.json
pennylane_linear_31q_8L_16gpu_mpi_final1.json
```

Historical filenames remain readable for provenance, but new figures and
reports must record the canonical name in their manifest rather than silently
renaming old payloads.

Canonical runtime variables use the `FQ_SV_` prefix. The previous
`FQ_STATEVECTOR_...` names remain accepted as compatibility aliases.
