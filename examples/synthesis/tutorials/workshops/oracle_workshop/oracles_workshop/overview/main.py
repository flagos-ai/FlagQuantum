"""Run all Oracle Workshop exercises and their full solution replays locally."""

from .workflow import classify_deutsch_jozsa, phase_table, sample, verify


def run_example() -> None:
    for index, name in enumerate(
        (
            "bit_oracle",
            "auxiliary_kickback",
            "direct_kickback",
            "deutsch_jozsa",
            "grover",
            "direct_kickback",
            "deutsch_jozsa",
            "grover",
        )
    ):
        check = verify(name)
        print(name, "Full-state error:", check.maximum_state_error)
        print("Dirty hidden workspace:", check.dirty_workspace_probability)
        print(check.program.report["resources"])
        if name in {"auxiliary_kickback", "direct_kickback"}:
            print(phase_table(check.program))
        elif name != "bit_oracle":
            observations = sample(check.program, shots=1000, seed=1970 + index)
            print(observations)
            if name == "deutsch_jozsa":
                print("The function is", classify_deutsch_jozsa(observations))


if __name__ == "__main__":
    run_example()
