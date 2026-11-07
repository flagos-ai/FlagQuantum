"""Control (X-3Z)/4 by a coherent two-bit word equal to 01.

Run python -m examples.algorithms.linear_combination_controls.
Add --stacked to extend an existing positive control.
"""

import argparse
from collections.abc import Callable, Mapping
from math import sqrt

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def main(*, stacked: bool = False) -> None:
    def block(label: str) -> UnitaryBlockEncoding:
        def emit(
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            bits: tuple[int, ...],
            inverse: bool,
        ) -> tuple[Instruction, ...]:
            return (Instruction(label, ports["q"]),)

        return UnitaryBlockEncoding((("q", 1),), emit)

    def factory(
        label: str,
    ) -> Callable[
        [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
        FieldOperation,
    ]:
        def bind(
            predicate: int,
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            bits: tuple[int, ...],
            inverse: bool,
        ) -> FieldOperation:
            return FieldOperation(
                max(predicate, *ports["system"]) + 1,
                (Instruction("c" + label, (predicate, *ports["system"])),),
            )

        return bind

    encoding = LinearCombinationBlockEncoding((block("x"), block("z")), (1.0, -3.0), 1)
    binders = (factory("x"), factory("z"))
    registers = (ControlRegister(2, (1,)),)
    psi = torch.tensor([1, 1j], dtype=torch.complex128) / sqrt(2)
    count = 2 + int(stacked)
    width = 11 + 2 * int(stacked)
    amplitudes = torch.complex(
        torch.arange(1, (1 << count) + 1).double(),
        torch.arange(1 << count).double(),
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    indices = [
        (word << (width - count)) + (bit << (width - count - 1))
        for word in range(1 << count)
        for bit in range(2)
    ]
    initial[indices] = (amplitudes[:, None] * psi).reshape(-1)
    system = (count,)
    ancilla = (count + 1,)
    resource = tuple(range(count + 2, count + 6))
    predicate = tuple(range(count + 6, count + 7 + int(stacked)))
    work = (predicate[-1] + 1, predicate[-1] + 2)
    current = (2,) if stacked else ()
    circuit = DynamicCircuit(width, inputs=initial, dtype=torch.complex128)
    circuit.append(
        encoding.make_on(
            binders,
            system=system,
            ancilla=ancilla,
            resource=resource,
            prepare_workspace=work,
            prepare_classical_bits=(len(predicate),),
            traversal_workspace=work[:1],
            traversal_classical_bits=(len(predicate),),
            workspaces=((), ()),
            classical_bits=((), ()),
            controls=((0, 1),),
            control_registers=registers,
            control_workspace=predicate,
            control_classical_bits=tuple(range(len(predicate))),
            existing_control=current,
        )
    )
    ir = circuit.to_ir()
    result = run_dynamic(circuit, shots=8, seed=71, strategy="batched")
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    active = ((x - 3 * z) @ psi) / 4
    expected = torch.cat(
        [
            amplitudes[word] * (active if word == (3 if stacked else 1) else psi)
            for word in range(1 << count)
        ]
    )
    torch.testing.assert_close(
        result.final_states[:, indices], expected.expand(8, -1), atol=4e-12, rtol=0
    )
    dirty = result.final_states.reshape(8, -1, 1 << (len(predicate) + len(work)))[
        :, :, 1:
    ]
    torch.testing.assert_close(dirty, torch.zeros_like(dirty), atol=4e-12, rtol=0)
    circuit.append(
        encoding.make_on(
            binders,
            system=system,
            ancilla=ancilla,
            resource=resource,
            prepare_workspace=work,
            prepare_classical_bits=(len(predicate),),
            traversal_workspace=work[:1],
            traversal_classical_bits=(len(predicate),),
            workspaces=((), ()),
            classical_bits=((), ()),
            controls=((0, 1),),
            control_registers=registers,
            control_workspace=predicate,
            control_classical_bits=tuple(range(len(predicate))),
            existing_control=current,
            adjoint=True,
        )
    )
    restored = run_dynamic(circuit, shots=8, seed=73, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=4e-12,
        rtol=0,
    )
    prep = encoding.coefficient_preparation.resource_model()
    select = encoding.select.resource_model(
        (ComponentCost("CX/CZ", 2, GateCost(clifford=1)),) * 2, controlled=True
    )
    report = estimate_ftqc_resources(
        encoding.resource_model(
            select, prep, prep, control_registers=registers, existing_control=stacked
        ),
        emitted_ir=ir,
    )
    assert report.emitted is not None
    print(
        "Activation word 01 and existing positive control; all eight words remain coherent"
        if stacked
        else "Activation word 01; all four control words remain coherent"
    )
    print(
        "Model T equivalents (abstract Fredkin)", report.logical.gates.total_t_count()
    )
    print("Emitted T/Tdag (exact Fredkin)", report.emitted.t_count)
    print("Model measurements", report.logical.gates.measurement)
    print("Model peak logical qubits", report.logical.peak_logical_qubits)
    print(
        "Projected controlled block, shared predicates and full complex inverse verified"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked", action="store_true")
    main(stacked=parser.parse_args().stacked)
