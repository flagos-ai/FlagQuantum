"""Bind the native unit-step permutation below the source atomic ModAddK boundary."""

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_gate_function import bit_gate_function
from ..component_wire_graph import wire_component_program
from .bounded_increment import bounded_increment_ir, bounded_increment_requirements


def bounded_increment_program(
    width: int,
    modulus: int,
    values: tuple[int, ...] = (),
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Increment an MSB-first unsigned word, fixing padding and inactive controls.

    Control elements follow values in order. Private work is clean on return.
    The frozen source supplies a permutation and five-Add model but no primitive
    decomposition. Descendants here retain the existing native physical recipe.
    """
    if type(width) is not int or width < 1:
        raise ValueError("Stored bounded increment requires a positive width")
    work, slots = bounded_increment_requirements(width, modulus, len(values))
    count = len(values)
    logical = count + width
    size = logical + work
    gates = bounded_increment_ir(
        tuple(range(count, logical)),
        modulus,
        tuple(range(logical, size)),
        tuple(range(slots)),
        controls=tuple(range(count)),
        control_values=values,
        adjoint=adjoint,
    )
    bit = quantum_bit_type()
    signature = Signature(
        (
            *((Register("ctrl", bit, (count,)),) if count else ()),
            Register("x", unsigned_register_type(width)),
        )
    )
    leaf = ProgramRecord(
        (
            bit_gate_function(
                "bounded_increment_gates",
                (Register("bits", bit, (size,)),),
                CircuitIR(size, gates),
            ),
        )
    )
    ports = {
        **({"ctrl": tuple(range(count))} if count else {}),
        "x": tuple(range(count, logical)),
    }
    return wire_component_program(
        signature, ports, ports, ((leaf, {"bits": tuple(range(size))}),)
    )
