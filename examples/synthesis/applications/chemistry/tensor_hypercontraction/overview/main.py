"""Compile the full tensor-hypercontraction resource experiment."""

import flagquantum as fq
from flagquantum.compiler.resource_estimation import estimate_resources

from . import model, quantum, verification
from .layout import Layout


def describe(data: model.Model) -> None:
    print(f"Precision and spin widths: {model.N_EPSILON}, {model.N_UD}")
    print(f"QROM rows: {len(data.mu_alt)}")
    print(f"RandomState(0) R_BR: {data.rotation:.15g}")
    print(f"Clock amplitude count: {len(data.clock)}")
    print(f"Clock leaves: {model.CLOCK_STEPS}")


def report(label: str, circuit: fq.Circuit) -> None:
    ir = circuit.to_ir()
    estimate = estimate_resources(ir)
    print(
        f"{label}: declared={estimate.n_qubits}, used={estimate.used_qubits}, "
        f"operations={estimate.n_operations}, depth={estimate.depth}"
    )


def report_checks(data: model.Model, layout: Layout) -> None:
    print(f"Verified emitted subcircuits: {verification.verify(data, layout)}")


def main() -> None:
    data = model.generate()
    describe(data)
    compiled = quantum.compile_all(data)
    report("PREPARE part 1", compiled.prepare_one)
    report("PREPARE part 2", compiled.prepare_two)
    report("SELECT", compiled.select)
    report("Full clock program", compiled.complete)
    report_checks(data, compiled.layout)


if __name__ == "__main__":
    main()
