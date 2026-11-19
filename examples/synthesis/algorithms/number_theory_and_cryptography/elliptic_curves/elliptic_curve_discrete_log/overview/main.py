"""Run the two native elliptic-curve workflows and report their resources."""

from collections.abc import Sequence

from flagquantum.compiler import ResourceEstimate, estimate_resources

from . import workflow


def print_resources(label: str, resources: ResourceEstimate) -> None:
    print(f"{label}: {resources.n_qubits} {resources.n_operations} {resources.depth}")


def print_frequencies(rows: Sequence[workflow.Frequency]) -> None:
    print("Observed outcomes:", len(rows))
    print("Observed draws:", sum(row.count for row in rows))
    for row in rows[:10]:
        print(row.left / 8, row.right / 8, row.point, row.count)


def print_logarithms(rows: Sequence[workflow.Frequency]) -> None:
    logs = [row.logarithm() for row in rows if row.logarithm() is not None]
    assert logs[:10] == [4] * 10
    print("Top ten eligible logarithms:", logs[:10])


def main() -> None:
    point, point_resources = workflow.sampled_point_add()
    assert point == (4, 5)
    print("Sampled point sum:", point)
    print_resources("Mock point gates", point_resources)
    rows, period_resources, mock_error = workflow.sampled_period_finding()
    print_resources("Mock ECDLP gates", period_resources)
    print("Mock absolute state error:", mock_error)
    print_frequencies(rows)
    print_logarithms(rows)

    point_circuit = workflow.point_add_circuit()
    print("Kaliski point sum:", workflow.checked_point_add(point_circuit))
    print_resources("Kaliski point gates", estimate_resources(point_circuit))
    del point_circuit
    period_circuit = workflow.period_finding_circuit()
    print_resources("Kaliski ECDLP gates", estimate_resources(period_circuit))
    print(
        "Kaliski absolute state error:", workflow.checked_period_finding(period_circuit)
    )


if __name__ == "__main__":
    main()
