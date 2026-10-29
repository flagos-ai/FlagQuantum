"""Private CS06-A: check an actual plan before running a routed phase example.

This expert seam demonstration does not change the normal synthesis facade or
select arithmetic candidates. Run ``python -m examples.synthesis_execution_contract``.
"""

import math

import torch

import flagquantum as fq
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler.coupling import CouplingMap
from flagquantum.compiler.layout import remove_layout_restore
from flagquantum.services._synthesis_execution import (
    ExecutionContractError,
    inspect_execution,
)


def main() -> None:
    topology = CouplingMap(3, ((0, 1), (1, 2)))
    logical = fq.Circuit(3, dtype=torch.complex128).h(0).s(0).cx(0, 2)
    routed = compile_program(
        logical,
        coupling_map=topology,
        routing_strategy="persistent_layout",
        optimize=False,
    )
    program = remove_layout_restore(routed)
    options = fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    plan = fq.plan(program, options=options)
    try:
        inspect_execution(plan, basis=("h", "s", "swap", "cx"), coupling=topology)
    except ExecutionContractError as error:
        assert error.code == "execution.basis"
        print("runtime basis drift rejected:", error.code)
    else:
        raise AssertionError("the source-only basis should not authorize u3 execution")

    # A new, explicit target request admits the actual canonical basis.
    checked = inspect_execution(
        plan,
        basis=("u3", "swap", "cx"),
        coupling=topology,
        limits=CircuitLimits(max_qubits=3, max_operations=3, max_depth=3),
        registers={"input": (0,), "result": (2,)},
    )
    result = fq.run(checked.plan)
    resources = checked.verify_result(result)
    state = result.statevector().reshape(-1)
    reference = torch.zeros_like(state)
    reference[0] = 1 / math.sqrt(2)
    reference[3] = torch.tensor(1j, dtype=state.dtype) / math.sqrt(2)
    torch.testing.assert_close(state, reference, atol=1e-12, rtol=0)
    assert checked.registers == {"input": (1,), "result": (2,)}
    print("verified physical registers:", dict(checked.registers))
    print(
        f"execution resources: qubits={resources.n_qubits} operations={resources.n_operations} depth={resources.depth}"
    )
    print("nonidentity routing and full complex phase: passed")
    print(
        "Inspection and result verification neither plan nor execute; fq.plan and fq.run own those steps."
    )


if __name__ == "__main__":
    main()
