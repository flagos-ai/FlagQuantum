"""Run all native QCE part I exercises and the complete source solution replays."""

from .workflow import STAGES, numeric_attributes, phase_table, resources, sample, verify


def run_example() -> None:
    stages = (*STAGES, *STAGES[1:])
    for index, name in enumerate(stages):
        check = verify(name)
        print(name, "Full-state error:", check.maximum_state_error)
        print("Dirty compiler workspace:", check.dirty_workspace_probability)
        print(resources(check))
        if name in {"integer_sum", "fractional_sum"}:
            print("Inferred c:", numeric_attributes(check.program, "c"))
        if name == "product_phase":
            print(phase_table(check.program))
        else:
            print(sample(check.program, shots=1000, seed=1970 + index))


if __name__ == "__main__":
    run_example()
