# Quafu explicit-mapping live execution evidence

> Commit references below have been mapped to the publication history.
> Recorded outcomes and approval status are unchanged.

On 2026-09-09, FlagQuantum completed one live Bell-circuit execution on the
Quafu `Dongling` QPU through the public `fq.run` path and the independently
installed QSteed compiler plugin.

The caller explicitly requested the ordered mapping `(3, 10)`. QSteed checked
that both physical qubits and their calibrated coupler existed in the current
snapshot, compiled the logical circuit, and returned the same mapping in the
`CircuitIR` execution target. FlagQuantum then submitted logical OpenQASM with
`compiler=None`. Quafu task `2609091513234674683` finished successfully and its
returned physical circuit contained `H Q3` followed by `Cnot(Q3, Q10)`.

The complete compact record is
[`quafu_explicit_mapping_dongling_20260909.json`](../../artifacts/development/quafu_explicit_mapping_dongling_20260909.json).
It connects:

1. FlagQuantum source revision `734cdeae7d380af1b15cf5c86911820f4c20c99b` and QSteed plugin revision `a514227`;
2. calibration timestamp and requested logical-to-physical mapping;
3. source and compiled IR identities;
4. submitted logical QASM and immutable deployment hashes;
5. provider task identity, returned physical circuit, and 1024-shot counts.

This is development evidence for the end-to-end execution contract. It is not
release certification, a performance result, or a general claim about QPU
quality. Quafu returns its final physical circuit after submission, so that
part of the chain remains post-execution diagnostic evidence rather than a
strict platform-side pre-submission circuit guarantee.

To reproduce the workflow, install the QSteed plugin, provide
`QUAFU_API_TOKEN` outside source control, choose physical qubits from the
current calibration snapshot, and run:

```python
result = fq.run(
    fq.Circuit(2).h(0).cx(0, 1),
    compiler="qsteed",
    target="quafu:Dongling",
    target_qubits=(3, 10),
    shots=1024,
    name="flagquantum_explicit_mapping_evidence",
)
```

Physical availability and calibration change over time. A later reproduction
must select a currently valid connected pair rather than assuming `(3, 10)` is
still available.
