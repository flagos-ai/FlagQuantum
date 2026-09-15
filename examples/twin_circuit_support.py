"""Check a circuit against topology-qualified Twin evidence without QPU I/O.

Run after ``examples/remote/quafu_twin_evidence.py`` has produced the two input
artifacts for a three-qubit mapping and adjust the declared mapping and circuit
to that prospectively validated workload.
"""

import flagquantum as fq


# In an application, load artifacts produced by a prospective validation run.
twin = fq.twin.load_twin("qpu-twin.json")
evidence = fq.twin.load_evidence("twin-evidence.json")

support = fq.twin.TwinCircuitSupport(
    evidence=evidence,
    directed_couplers=((20, 27), (27, 20), (27, 34), (34, 27)),
    maximum_circuit_depth=8,
)

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
report = support.evidence_report(twin, circuit)

print(report.status)
print(report.tv_error_bound)
print(report.confidence_level)
