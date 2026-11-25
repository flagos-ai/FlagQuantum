"""Run the native MCX tutorial with local compilation evidence."""

from flagquantum.synth import Program

from .workflow import (
    BostonCircuit,
    CXCandidate,
    boston_circuit,
    check_50,
    cx_search,
    program_14,
    resources,
    verify,
    verify_program,
)


def report_depth(program: Program) -> None:
    cost = resources(program)
    print("14-control depth circuit:", cost.n_qubits, cost.n_operations, cost.depth)
    print("14-control absolute state check:", verify_program(program))


def report_boston(case: BostonCircuit) -> None:
    print("Boston routed device qubits:", case.physical)
    print("Boston offline candidate depths:", case.candidate_depths)
    cost = case.resources
    print("Boston selected circuit:", cost.n_qubits, cost.n_operations, cost.depth)
    print("Boston selected CZ count:", cost.operation_counts["cz"])
    print(
        "Boston state check modulo global phase:",
        verify(case.circuit, case.logical, projective=True),
    )


def report_cx(candidates: tuple[CXCandidate, ...], selected: CXCandidate) -> None:
    rows = []
    for candidate in candidates:
        cost = candidate.resources
        check = verify(candidate.circuit, candidate.logical)
        rows.append(
            (
                candidate.label,
                cost.n_qubits,
                cost.operation_counts["cx"],
                cost.depth,
                check.error,
            )
        )
    print("17-qubit-cap CX candidates:", tuple(rows))
    print("17-qubit-cap selected CX:", selected.resources.operation_counts["cx"])


def report_large() -> None:
    for controls, target in ((0, 0), (49, 0), (50, 0), (50, 1)):
        cost, observed = check_50(controls, target)
        print(
            f"50-control target with {controls} enabled and target {target}:", observed
        )
        if controls == target == 0:
            print(
                "50-control native circuit:",
                cost.n_qubits,
                cost.n_operations,
                cost.depth,
            )


def main() -> None:
    report_depth(program_14(20, 5))
    report_boston(boston_circuit())
    report_cx(*cx_search())
    report_large()


if __name__ == "__main__":
    main()
