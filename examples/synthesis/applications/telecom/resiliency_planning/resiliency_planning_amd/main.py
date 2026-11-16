"""The retained AMD-named source has the same hardware-neutral quantum program."""

import flagquantum as fq
from examples.synthesis.applications.telecom.resiliency_planning.main import (
    build_program,
    main,
)

__all__ = ["build_program", "main"]

if __name__ == "__main__":
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())
