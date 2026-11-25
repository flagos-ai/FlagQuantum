"""Run both source-size discrete-log experiments and exact classical recovery."""

from .workflow import illustrative_curves, run_estimate


def main() -> None:
    """Print actual samples, source-selected rows, resources and illustration sizes."""
    for modulus in (5, 13):
        result = run_estimate(modulus, seed=1600 + modulus)
        print(
            "Modulus:", modulus, "Shots:", result.shots, "Logarithm:", result.logarithm
        )
        print("Leading selected rows:", result.candidates[:10])
        print(
            "Full-state error:",
            result.maximum_state_error,
            "Dirty work:",
            result.dirty_workspace_probability,
        )
        print(result.resources)
    for bits in (5, 8):
        curves = illustrative_curves(bits)
        print(
            "Source illustrative size:",
            len(curves.alpha),
            "Real log ratio:",
            curves.real_log_ratio,
        )


if __name__ == "__main__":
    main()
