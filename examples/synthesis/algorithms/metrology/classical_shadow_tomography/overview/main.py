"""Run the source-size native Pauli and Clifford classical-shadow workflow."""

from .workflow import run_workflow


def main() -> None:
    """Print actual reconstruction, expectation estimates and sample bounds."""
    workflow = run_workflow()
    for name, shadow in (("Pauli", workflow.pauli), ("Clifford", workflow.clifford)):
        print(name, "single shots:", len(shadow.bits))
        print("First ten logical bit rows:", shadow.bits[:10].tolist())
        print("Reconstruction matrix:", shadow.density)
        print("Frobenius error / ZZ:", shadow.frobenius_error, shadow.zz_estimate)
        print(
            "Operation range:",
            min(r.n_operations for r in shadow.resources),
            max(r.n_operations for r in shadow.resources),
        )
    print(
        "Required Pauli / ideal-Clifford samples:",
        workflow.pauli_bound,
        workflow.clifford_bound,
    )
    print(
        "Bounds are computed, not executed; the finite-depth Clifford walk is approximate."
    )


if __name__ == "__main__":
    main()
