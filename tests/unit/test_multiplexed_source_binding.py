"""Replace actual multiplexed source objects with native typed-layout emitters."""

from collections import Counter
from dataclasses import replace
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.compiler.registers.lookup.multiplexed_swap import MultiplexedSwapLayout
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_swap_network_file import source_network as source_network

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def logical_operator(ir: CircuitIR, logical: int) -> torch.Tensor:
    """Prove raw-history identity and final cleanup, then return the full operator."""
    count = certify_erasure_histories(
        ir.instructions,
        n_wires=ir.n_wires,
        zero_wires=tuple(range(logical, ir.n_wires)),
        check_clean_output=False,
    )
    stride = 1 << (ir.n_wires - logical)
    columns = []
    for column in range(1 << logical):
        initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
        initial[column * stride] = 1
        result = forced_branch(initial, ir.instructions, (0,) * count) * sqrt(2**count)
        clean = torch.zeros_like(result)
        clean[::stride] = result[::stride]
        torch.testing.assert_close(result, clean, atol=8e-12, rtol=0)
        columns.append(result[::stride])
    return torch.stack(columns, dim=1)


@pytest.mark.parametrize(
    "widths,shape,controls",
    [
        ((2,), (3,), 0),
        ((2,), (3,), 1),
        ((2,), (3,), 2),
        ((1, 1), (2, 2), 1),
    ],
)
def test_resolved_multiplexed_source_full_operators_and_native_replacement(
    source_network: Any, widths: tuple[int, ...], shape: tuple[int, ...], controls: int
) -> None:
    from attrs import evolve
    from qualtran import BQUInt, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap
    from sympy import symbols

    sw = symbols(f"s:{len(widths)}", positive=True, integer=True)
    ns = symbols(f"N:{len(shape)}", positive=True, integer=True)
    b = symbols("b", positive=True, integer=True)
    declaration = MultiplexedCSwap(
        tuple(
            SourceRegister(f"address{i}", BQUInt(w, n))
            for i, (w, n) in enumerate(zip(sw, ns, strict=True))
        ),
        b,
    )
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(declaration)
    source = evolve(
        declaration,
        selection_regs=tuple(
            SourceRegister(f"address{i}", BQUInt(w, n))
            for i, (w, n) in enumerate(zip(widths, shape, strict=True))
        ),
        target_bitsize=1,
        control_regs=(
            ()
            if not controls
            else (SourceRegister("enable", QBit(), shape=(controls,)),)
        ),
    )
    native = MultiplexedSwapLayout(
        tuple(
            Register(f"address{i}", bounded_unsigned_register_type(w, n))
            for i, (w, n) in enumerate(zip(widths, shape, strict=True))
        ),
        1,
        (
            ()
            if not controls
            else (Register("enable", quantum_bit_type(), shape=(controls,)),)
        ),
    )
    assert_source_model(native.resource_model(), source)
    ports = tuple((r.name, int(r.total_bits())) for r in source.signature)
    logical = sum(bits for _, bits in ports)
    work, slots = native.requirements()
    original = import_register_emitter(source)
    imported = replace(
        original,
        snapshots=(
            CircuitIR.from_json(original.snapshots[0].to_json()),
            CircuitIR.from_json(original.snapshots[1].to_json()),
        ),
    )
    # Replace the implementation at the same named consumer, retaining only IR.
    native_emitter = CircuitRegisterEmitter(
        ports,
        (
            CircuitIR(
                logical + work,
                native.bind(
                    tuple(range(logical)),
                    tuple(range(logical, logical + work)),
                    tuple(range(slots)),
                ),
            ),
            CircuitIR(
                logical + work,
                native.bind(
                    tuple(range(logical)),
                    tuple(range(logical, logical + work)),
                    tuple(range(slots)),
                    adjoint=True,
                ),
            ),
        ),
    )
    matrices = []
    streams = []
    for emitter in (imported, native_emitter):
        private, results = emitter.requirements()
        bound = BlackBoxPrepare(ports, (), emitter)
        directions = []
        for adjoint in (False, True):
            ir = (
                Circuit(logical + private)
                .append(
                    bound.make_on(
                        selection=tuple(range(logical)),
                        junk=(),
                        workspace=tuple(range(logical, logical + private)),
                        classical_bits=tuple(range(results)),
                        adjoint=adjoint,
                    )
                )
                .to_ir()
            )
            assert isinstance(ir, CircuitIR)
            streams.append(ir)
            counts = Counter(g.name for g in ir.instructions)
            if len(shape) == 1:
                assert len(ir.instructions) == (73, 91, 108)[controls]
                assert counts["measure"] == controls + 1
                assert counts["t"] + counts["tdg"] == 25 + 4 * controls
            directions.append(logical_operator(ir, logical))
        forward, backward = directions
        torch.testing.assert_close(
            backward @ forward,
            torch.eye(1 << logical, dtype=torch.complex128),
            atol=1e-11,
            rtol=0,
        )
        matrices.extend(directions)
    for source_ir, native_ir in zip(streams[:2], streams[2:], strict=True):
        assert Counter(g.name for g in source_ir.instructions) == Counter(
            g.name for g in native_ir.instructions
        )
    torch.testing.assert_close(matrices[0], matrices[2], atol=8e-12, rtol=0)
    torch.testing.assert_close(matrices[1], matrices[3], atol=8e-12, rtol=0)


@pytest.mark.parametrize("values", [0, 1, (0, 1)])
def test_outer_controlled_and_remains_an_explicit_source_boundary(
    source_network: Any, values: int | tuple[int, ...]
) -> None:
    from qualtran import BQUInt, CtrlSpec
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap

    source = MultiplexedCSwap(SourceRegister("selection", BQUInt(2, 3)), 1).controlled(
        CtrlSpec(cvs=values)
    )
    with pytest.raises(
        ValueError,
        match="Foreign oracle primitive cannot be decomposed: Controlled.*And",
    ):
        import_register_emitter(source)


@pytest.mark.parametrize("activation", [(0,), (1,), (0, 1)])
def test_supported_generic_control_actual_source_operators(
    source_network: Any, activation: tuple[int, ...]
) -> None:
    from qualtran import BQUInt, CtrlSpec
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap

    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.lookup.multiplexed_swap import (
        control_multiplexed_cswap_ir,
    )

    base = MultiplexedCSwap(SourceRegister("selection", BQUInt(1, 2)), 1)
    source = base.controlled(
        CtrlSpec(cvs=activation[0] if len(activation) == 1 else activation)
    )
    captured = import_register_emitter(source)
    emitter = replace(
        captured,
        snapshots=(
            CircuitIR.from_json(captured.snapshots[0].to_json()),
            CircuitIR.from_json(captured.snapshots[1].to_json()),
        ),
    )
    outer = len(activation)
    logical = outer + 4
    private, slots = emitter.requirements()
    ports = tuple((r.name, int(r.total_bits())) for r in source.signature)
    bound = BlackBoxPrepare(ports, (), emitter)
    operators = []
    for adjoint in (False, True):
        ir = (
            Circuit(logical + private)
            .append(
                bound.make_on(
                    selection=tuple(range(logical)),
                    junk=(),
                    workspace=tuple(range(logical, logical + private)),
                    classical_bits=tuple(range(slots)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert isinstance(ir, CircuitIR)
        gates = control_multiplexed_cswap_ir(
            (tuple(range(outer)),),
            (ControlRegister(1, activation),),
            (),
            ((outer,),),
            (2,),
            ((outer + 1,), (outer + 2,)),
            (outer + 3,),
            tuple(range(logical, logical + outer)),
            tuple(range(outer)),
            adjoint=adjoint,
        )
        native = CircuitIR(logical + outer, gates)
        assert Counter(g.name for g in ir.instructions) == Counter(
            g.name for g in native.instructions
        )
        original = logical_operator(ir, logical)
        torch.testing.assert_close(
            original, logical_operator(native, logical), atol=8e-12, rtol=0
        )
        operators.append(original)
    torch.testing.assert_close(
        operators[1] @ operators[0],
        torch.eye(1 << logical, dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
