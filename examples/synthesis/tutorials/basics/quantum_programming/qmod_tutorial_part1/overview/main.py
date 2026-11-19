"""Run all native tutorial solutions and the expected ownership refusal."""

from .workflow import initialization_refusal, sample, signed_distribution, verify


def run_example() -> None:
    print(initialization_refusal())
    for name in (
        "bell",
        "ghz",
        "signed_ghz",
        "distributions",
        "corrected_initialization",
        "rotations",
        "parameterized",
        "layer",
        "arithmetic",
        "accumulation",
        "conditional",
    ):
        row = verify(name)
        print(name, row.maximum_error, row.dirty_workspace_probability)
        print(row.program.report["resources"])
        print(sample(row.program, shots=10000 if name == "distributions" else 1000))
        if name == "signed_ghz":
            print(dict(signed_distribution(row.program)))


if __name__ == "__main__":
    run_example()
