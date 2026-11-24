"""Compare typed indexed calls and exact inverses with original source loops."""

from math import pi

import pytest
import torch

from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.indexed_selection_component import (
    indexed_selection_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def positive_gate_pairs() -> tuple[tuple[ProgramRecord, ProgramRecord], ...]:
    """Keep controlled T as its original atomic controlled-phase boundary."""
    factories = fixed_gate_factories()
    forward = bit_gate_function(
        "atomic_t",
        tuple(Signature.build(ctrl=1, q=1)),
        CircuitIR(2, (Instruction("cphase", (0, 1), params={"theta": pi / 4}),)),
    )
    inverse = bit_gate_function(
        "atomic_t_inverse",
        tuple(Signature.build(ctrl=1, q=1)),
        CircuitIR(2, (Instruction("cphase", (0, 1), params={"theta": -pi / 4}),)),
    )
    pairs = [(ProgramRecord((forward,)), ProgramRecord((inverse,)))]
    for suffix in ("hadamard.CHadamard", "z_basis.CZ", "cnot.CNOT"):
        function = factories["qualtran.bloqs.basic_gates." + suffix](
            suffix.rsplit(".", 1)[-1], None, {}
        )
        assert isinstance(function, FunctionRecord)
        record = ProgramRecord((function,))
        pairs.append((record, record))
    return tuple(pairs)


@pytest.mark.parametrize("shape", ((4,), (3,), (2, 2)))
@pytest.mark.parametrize("control_value", (None, 0, 1))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_indexed_calls_full_complex_operators_and_work(
    shape: tuple[int, ...], control_value: int | None, inverse: bool
) -> None:
    from importlib.metadata import version

    import kickmix
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import Hadamard, TGate, XGate, ZGate
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    np = pytest.importorskip("numpy")
    assert (
        version("qualtran") == "0.7.0"
        and kickmix.__version__ == "0.1"
        and hasattr(kickmix, "Simulator")
    )
    count = int(np.prod(shape))
    source = ApplyLthBloq(
        np.array((TGate(), Hadamard(), ZGate(), XGate())[:count], dtype=object).reshape(
            shape
        ),
        control_val=control_value,
    )
    selections = tuple(
        Register(
            "selection" if len(shape) == 1 else f"selection{i}",
            bounded_unsigned_register_type((size - 1).bit_length(), size),
        )
        for i, size in enumerate(shape)
    )
    control = (
        Register("control", quantum_bit_type()) if control_value is not None else None
    )
    pairs = positive_gate_pairs()[:count]
    record = indexed_selection_component_program(
        selections,
        (Register("q", quantum_bit_type()),),
        shape,
        tuple(pair[0] for pair in pairs),
        inverses=tuple(pair[1] for pair in pairs),
        control=control,
        adjoint=inverse,
    )
    if inverse:
        source = Adjoint(source)
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    inputs = []
    offset = 0
    for register in record.functions[0].registers or ():
        width = int(register.total_bits())
        inputs.append(tuple(range(offset, offset + width)))
        offset += width
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(offset, offset + 4))
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, offset),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=6e-11,
        rtol=0,
    )
