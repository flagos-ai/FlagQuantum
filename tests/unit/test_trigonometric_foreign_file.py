"""Original trigonometric libraries and SDK tensor consumers retain boundaries."""

from collections import Counter
from collections.abc import Mapping
from itertools import product

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.fixed_point.arcsine_declaration import (
    ArcsineDeclaration,
)
from flagquantum.compiler.registers.lookup.arctan_lookup import (
    arctan_lookup_ir,
    arctan_lookup_requirements,
)
from flagquantum.compiler.registers.lookup.arctangent_declaration import (
    ArctangentDeclaration,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_import import (
    import_cirq_register_gate,
    import_cirq_register_optree,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
cirq = pytest.importorskip("cirq")
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("kind", ("arcsine", "arctangent"))
@pytest.mark.parametrize("symbolic", (False, True))
def test_complete_actual_library_depths_generic_arguments_and_models(
    kind: str, symbolic: bool
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcSin, ArcTan
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    n = sympy.Symbol("n") if symbolic else 10
    source = ArcSin(n, 7) if kind == "arcsine" else ArcTan(4, n)
    for depth in (0, 1, 2):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        assert bloqs_from_proto(proto)[0] == source
        assert len(library.entries) == (1 if depth == 0 else 2)
        if kind == "arctangent" and depth:
            assert [count for _, count in library.entries[0].counts] == [
                sympy.Symbol("_c0")
            ]
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString()
    )

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        assert registers is not None
        values = {}
        for key, value in arguments.items():
            assert isinstance(value, (int, sympy.Expr))
            values[key] = value
        if kind == "arcsine":
            declaration: ArcsineDeclaration | ArctangentDeclaration = (
                ArcsineDeclaration(
                    values["bitsize"],
                    values["num_frac"],
                    values["num_iters"],
                    values["degree"],
                )
            )
        else:
            declaration = ArctangentDeclaration(
                values["selection_bitsize"], values["target_bitsize"]
            )
        assert tuple(declaration.signature) == registers
        return FunctionRecord(name, (), registers=registers)

    resolved = resolve_component_program(
        library, 0, {library.entries[0].declaration.name: factory}
    )
    assert resolved.functions[0].registers == tuple(native_signature(source.signature))


@pytest.mark.parametrize("fraction", (2, 3, 4))
def test_arcsine_preserves_original_missing_physical_body_and_scalar_overflow(
    fraction: int,
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.trigonometric import ArcSin

    from flagquantum.simulation.classical_arcsine import source_arcsine_classical_values

    source = ArcSin(4, fraction)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    assert source.on_classical_vals(x=15, result=0) == {"x": 15, "result": 19}
    assert source_arcsine_classical_values(15, 0, 4) == (15, 19)
    ports = {
        "x": np.array(cirq.LineQubit.range(4)),
        "result": np.array(cirq.LineQubit.range(4, 8)),
    }
    operation, _ = source.as_cirq_op(cirq.SimpleQubitManager(), **ports)
    assert (
        import_cirq_register_gate(
            operation, signature=native_signature(source.signature)
        )
        is None
    )
    with pytest.raises(ValueError, match="unsupported_operation"):
        import_cirq_register_optree(
            [operation],
            signature=native_signature(source.signature),
            in_quregs=ports,
            out_quregs=ports,
        )


@pytest.mark.parametrize("left,right", ((1, 2), (2, 3), (3, 3)))
def test_actual_arctangent_sdk_tensor_and_owned_qrom_full_complex_histories(
    left: int, right: int
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.trigonometric import ArcTan
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = ArcTan(left, right)
    logical = left + 1 + right
    qubits = cirq.LineQubit.range(logical)
    ports = {
        "select": np.array(qubits[:left]),
        "sign": np.array(qubits[left : left + 1]),
        "target": np.array(qubits[left + 1 :]),
    }
    operation = source.on(*qubits)
    signature = native_signature(source.signature)
    assert import_cirq_register_gate(operation, signature=signature) is None
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    imported = import_cirq_register_optree(
        [operation], signature=signature, in_quregs=ports, out_quregs=ports
    )
    bound = bind_register_program(
        imported, (tuple(range(left)), (left,), tuple(range(left + 1, logical)))
    )
    assert Counter(g.name for g in bound.circuit.instructions) == {"any": 1}
    work, slots = arctan_lookup_requirements(left, right)
    gates = arctan_lookup_ir(
        tuple(range(left)),
        left,
        tuple(range(left + 1, logical)),
        tuple(range(logical + 1, logical + 1 + work)),
        tuple(range(slots)),
    )
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + work), dtype=torch.complex128)
    initial[:: 1 << work] = amplitudes
    expected = torch.zeros_like(initial)
    expected[:: 1 << work] = (
        torch.as_tensor(cirq.unitary(source), dtype=torch.complex128)
        @ amplitudes.reshape(-1, 2)
    ).flatten()
    sdk = DynamicCircuit(logical + 1 + work, inputs=initial, dtype=torch.complex128)
    sdk.append(FieldOperation(logical + 1 + work, bound.circuit.instructions))
    torch.testing.assert_close(
        forced_branch(initial, sdk.to_ir().instructions, ()),
        expected,
        rtol=0,
        atol=3e-12,
    )
    circuit = DynamicCircuit(logical + 1 + work, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(logical + 1 + work, gates))
    counts = Counter(g.name for g in gates)
    rows = tuple(source.apply(x, 0, 0) for x in range(1 << left))
    qrom = QROM(
        (np.array([r[1] for r in rows]), np.array([r[2] for r in rows])),
        (left,),
        (1, right),
        ((), ()),
    )
    model = get_cost_value(qrom, QECGatesCost())
    assert model.total_t_count() == counts["t"] + counts["tdg"]
    assert model.measurement == counts["measure"]
    # This is the QROM execution model. ArcTan declares a separate symbolic product estimate.
    assert source.t_complexity().t == 8 * sympy.Symbol("_c0") * right**2
    for history in product((0, 1), repeat=counts["measure"]):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, expected / 2 ** (len(history) / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(branch.reshape(-1, 1 << work)[:, 1:]) == 0
