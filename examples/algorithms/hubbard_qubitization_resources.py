"""Reproduce the concrete and symbolic Hubbard notebook resource queries."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's hubbard_model.ipynb and select_hubbard.ipynb.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization

from argparse import ArgumentParser
from dataclasses import asdict
from functools import partial
from typing import Literal

from flagquantum.algorithms.primitives.hubbard_prepare import (
    PrepareHubbard,
    hubbard_selection_layout,
)
from flagquantum.algorithms.primitives.hubbard_qubitization_resource_model import (
    source_hubbard_majorana_formula,
    source_hubbard_spin_up_formula,
)
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt, ceil, log2
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.hubbard_symbols import hubbard_consumer_wire_symbol
from flagquantum.drawer.register_score import (
    RegisterScore,
    RegisterStep,
    layout_register_score,
)
from flagquantum.drawer.register_score_data import MusicalScoreData, register_score_data
from flagquantum.qec.resources.estimation import (
    estimate_ftqc_resources,
    estimate_symbolic_ftqc_resources,
)


def notebook_models() -> tuple[tuple[int, ComponentCost], ...]:
    """Return the concrete top-level models used in the locked notebooks."""
    return (
        (4, SelectHubbard(4).source_resource_model()),
        (4, PrepareHubbard(4, t=1.0, u=4.0).source_resource_model()),
        (11, SelectHubbard(11, controlled=True).source_resource_model()),
    )


def notebook_signature(
    side: int, *, prepare: bool = False, controlled: bool = False
) -> Signature:
    """Describe the concrete notebook's through ports and coordinate extents."""
    control = (Register("control", quantum_bit_type()),) if controlled else ()
    selection = tuple(
        Register(
            name,
            bounded_unsigned_register_type(
                width, side if name in ("p_x", "p_y", "q_x", "q_y") else 2
            ),
        )
        for name, width in hubbard_selection_layout(side)
    )
    payload = (
        Register("temp", opaque_register_type(2))
        if prepare
        else Register("target", opaque_register_type(2 * side**2))
    )
    return Signature((*control, *selection, payload))


def notebook_consumer_signature(
    side: SymbolicInt, *, majorana: bool, controlled: bool = False
) -> Signature:
    """Describe the symbolic or concrete consumer's original register order."""
    control = (Register("control", quantum_bit_type()),) if controlled else ()
    bit = bounded_unsigned_register_type(1, 2)
    coordinate = bounded_unsigned_register_type(ceil(log2(side)), side)
    sites = (Register("x", coordinate), Register("y", coordinate))
    selection = (
        (*sites, Register("spin", bit)) if majorana else (Register("V", bit), *sites)
    )
    return Signature(
        (*control, *selection, Register("target", opaque_register_type(2 * side**2)))
    )


def notebook_consumer_score(
    side: int, *, majorana: Literal["X", "Y"] | None = None, controlled: bool = False
) -> MusicalScoreData:
    """Draw the original concrete Majorana or spin-up Z leaf."""
    ports = tuple(
        notebook_consumer_signature(
            side, majorana=majorana is not None, controlled=controlled
        )
    )
    widths = {port.name: int(port.bitsize) for port in ports}
    names = tuple(widths)
    step = RegisterStep(
        "Hubbard consumer", names, names, input_registers=ports, output_registers=ports
    )
    score = layout_register_score(
        widths, names, (step,), input_registers=ports, output_registers=ports
    )
    return register_score_data(
        score,
        register_wire_symbol=partial(hubbard_consumer_wire_symbol, majorana=majorana),
    )


def notebook_controlled_select_score() -> RegisterScore:
    """Show each source call and its logical operands for 11×11 SELECT."""
    signature = notebook_signature(11, controlled=True)
    _, model = notebook_models()[-1]
    widths = {register.name: int(register.bitsize) for register in signature}
    x_swap = ("V", "p_x", "q_x")
    y_swap = ("V", "p_y", "q_y")
    spin_swap = ("V", "alpha", "beta")
    operands = (
        ("control", "p_x", "p_y", "alpha", "target"),
        x_swap,
        y_swap,
        spin_swap,
        ("control", "q_x", "q_y", "beta", "target"),
        spin_swap,
        y_swap,
        x_swap,
        ("control",),
        ("control", "U"),
        ("control", "V", "q_x", "q_y", "target"),
    )
    names = (
        call.component.name for call in model.calls for _ in range(call.repetitions)
    )
    return layout_register_score(
        widths,
        tuple(widths),
        tuple(
            RegisterStep(name, ports, ports)
            for name, ports in zip(names, operands, strict=True)
        ),
        input_registers=tuple(signature),
        output_registers=tuple(signature),
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbolic",
        action="store_true",
        help="Also show symbolic controlled leaf models",
    )
    args = parser.parse_args()
    for side, model in notebook_models():
        print(f"{side}×{side} {model.name}")
        graph = modeled_call_graph(model, max_depth=1)
        for child, count in graph.leaf_counts.items():
            cost = estimate_ftqc_resources(child).logical.gates
            categories = {name: value for name, value in asdict(cost).items() if value}
            print(f"  {count} × {child.name}: {categories}")
        result = estimate_ftqc_resources(model).logical
        print("  Total logical gates:", asdict(result.gates))
        print("  Peak logical qubits:", result.peak_logical_qubits)
    if args.symbolic:
        from sympy import Symbol

        side = Symbol("n")
        for name, formula in (
            (
                "Controlled Majorana X",
                source_hubbard_majorana_formula(side, controlled=True),
            ),
            (
                "Controlled spin-up Z",
                source_hubbard_spin_up_formula(side, controlled=True),
            ),
        ):
            report = estimate_symbolic_ftqc_resources(formula)
            print(name)
            print("  Logical resources:", asdict(report.logical))
            print("  T equivalents:", report.t_equivalents)


if __name__ == "__main__":
    main()
