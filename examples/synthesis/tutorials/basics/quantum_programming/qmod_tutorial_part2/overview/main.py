"""Run the complete second native quantum programming tutorial."""

from . import quantum, reference, workflow


def main() -> None:
    for label, entry, expected, names, width, workspace, angles in (
        (
            "Product phase",
            "product_phase",
            reference.phase_state(),
            ("x", "y"),
            4,
            0,
            True,
        ),
        (
            "Predicate oracle",
            "solution_oracle",
            reference.phase_state(oracle=True),
            ("a", "b"),
            4,
            0,
            False,
        ),
        (
            "Local comparison",
            "local_comparison",
            reference.comparison_state(),
            ("x", "res"),
            7,
            4,
            False,
        ),
        (
            "Fourier addition",
            "fourier_add",
            reference.addition_state(controlled=False),
            ("x", "y"),
            7,
            0,
            True,
        ),
        (
            "Controlled Fourier addition",
            "controlled_fourier_add",
            reference.addition_state(controlled=True),
            ("ctrl", "x", "y"),
            8,
            0,
            False,
        ),
        (
            "Suzuki–Trotter",
            "pauli_evolution",
            reference.trotter_state(),
            ("qarr",),
            4,
            0,
            False,
        ),
    ):
        program, result, checked = workflow.run_native(
            entry,
            expected,
            names,
            max_qubits=width,
            max_workspace=workspace,
            angles=angles,
        )
        print(label, program.report["resources"])
        print(
            f"Full-state error {checked.error:.3g}, workspace leakage {checked.workspace_probability:.3g}"
        )
        if label in ("Product phase", "Predicate oracle"):
            print(checked.vector.tolist())
        else:
            print(workflow.read_counts(program, result, names, seed=16))
    print("X(0) + X(1)", workflow.observe_plus())
    print("Callback", workflow.read_callback())
    phase, checked = workflow.read_qpe()
    print("Phase estimate", dict(phase), "full-state error", checked.error)
    template = quantum.parameterized_ansatz()
    print("Symbolic template", template.parameter_names)
    print("Two finite sampling runs", workflow.read_bindings(template))
    print("Bound expectation", workflow.observe_binding(template))


if __name__ == "__main__":
    main()
