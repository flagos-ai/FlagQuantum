"""Close actual Hamiltonian controls through owned GQSP and Always boundaries."""

import sys
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any
from unittest.mock import patch

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    import cirq as cirq
    import numpy as np

    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_arguments import ControlSpecification
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_controlled_factories import (
    controlled_via_and_factories,
)
from flagquantum.compiler.formats.legacy_hamiltonian_gqsp_factories import (
    HamiltonianGQSPContext,
    controlled_hamiltonian_gqsp_factories,
)
from flagquantum.compiler.formats.legacy_qubitization_walk_factories import (
    ExactPair,
    WalkContext,
)
from flagquantum.compiler.formats.legacy_su2_factories import _positive_su2_program
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_control import _stored_control_adder
from flagquantum.compiler.registers.phase.generalized_qsp_component import (
    controlled_gqsp_component_program,
)
from flagquantum.compiler.registers.preparation.qubitization_walk_component import (
    doubly_controlled_select_walk_pair,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem._default_linear_combination_oracles import _signature
from flagquantum.ecosystem._generalized_qsp_component_library import (
    generalized_qsp_library_factories,
)
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.numerics.hamiltonian_qsp_coefficients import (
    hamiltonian_qsp_coefficients,
)
from flagquantum.simulation.numerics.qsp_complement import qsp_complementary_polynomial
from flagquantum.simulation.numerics.qsp_phases import qsp_phase_factors
from flagquantum.simulation.value_dimensions import SymbolicArrayShape
from tests.unit.test_chebyshev_component import assert_all_small_raw_histories
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_controlled_su2_source import locked_su2_source as locked_su2_source
from tests.unit.test_hubbard_stored_graph import source_terminals
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_legacy_block_encoding_factories import _oracles
from tests.unit.test_legacy_phased_block_factories import scalar_terminals
from tests.unit.test_legacy_qubitization_walk_factories import context_fixture
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def locked_hamiltonian_source() -> None:
    from hashlib import sha1
    from pathlib import Path

    from qualtran.bloqs.hamiltonian_simulation import hamiltonian_simulation_by_gqsp

    for filename, digest in (
        (
            "hamiltonian_simulation_by_gqsp.py",
            "21c3ee41f15e45391411ff665b70b7d3e8dec878",
        ),
        (
            "hamiltonian_simulation_by_gqsp_test.py",
            "37ecf029009ee7b97add54d60c5f9ecf433df11c",
        ),
    ):
        raw = (
            Path(hamiltonian_simulation_by_gqsp.__file__).parent / filename
        ).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def contexts(
    source: Any, walk_context: WalkContext, selection: ExactPair, preparation: ExactPair
) -> tuple[HamiltonianGQSPContext, HamiltonianGQSPContext]:
    ports = (Register("signal", quantum_bit_type()), *tuple(walk_context.signature))
    coefficients = hamiltonian_qsp_coefficients(
        source.alpha, source.t, source.precision
    )
    if isinstance(coefficients, SymbolicArrayShape):
        ordinary = ProgramRecord(
            (FunctionRecord("symbolic_gqsp", (), registers=ports),)
        )
        controlled = controlled_gqsp_component_program(ordinary, {})
    else:
        walk = walk_context.pair()
        positive = walk_context.pair(controlled=True, source_control_recipes=True)
        assert walk_context.positive_reflection_pair is not None
        assert walk_context.block_pair[0].functions[0].registers is not None
        double = doubly_controlled_select_walk_pair(
            walk_context.signature,
            Signature(walk_context.block_pair[0].functions[0].registers),
            selection,
            walk_context.positive_reflection_pair[0],
            select_control="control",
            reflection_control="control",
        )
        left = coefficients
        right = tuple(
            map(
                complex,
                qsp_complementary_polynomial(left, verify=True, verify_precision=1e-4),
            )
        )
        degree = (len(left) - 1) // 2
        synthesis_precision = source.precision / 2
        theta, phi, lambd = qsp_phase_factors(left, right)
        factory = next(
            iter(
                generalized_qsp_library_factories(
                    child_owner=lambda _: walk, positive_child_owner=lambda _: positive
                ).values()
            )
        )
        candidate = factory(
            "main",
            None,
            {
                "U": walk[0].functions[0],
                "P": left,
                "Q": right,
                "negative_power": degree,
                "precision": synthesis_precision,
            },
        )
        assert isinstance(candidate, ProgramRecord)
        ordinary = candidate
        forward = _rename_program(positive[0], "mixed_positive_walk")
        via = controlled_via_and_factories(
            control_system=lambda _: (
                forward,
                _stored_control_adder(forward.functions[0]),
            )
        )
        mixed = via["qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"](
            "mixed",
            None,
            {
                "subbloq": walk[0].functions[0],
                "ctrl_spec": ControlSpecification(
                    (
                        (quantum_bit_type(), np.array(1)),
                        (quantum_bit_type(), np.array(0)),
                    )
                ),
            },
        )
        assert isinstance(mixed, ProgramRecord)
        rotations = tuple(
            _positive_su2_program(
                "positive_" + str(i),
                {
                    "theta": float(t),
                    "phi": float(p),
                    "lambd": float(lambd) if i == 0 else 0,
                    "eps": synthesis_precision / (len(left) + 1),
                },
            )
            for i, (t, p) in enumerate(zip(theta, phi, strict=True))
        )
        remaining = degree
        controlled_calls = {}
        for index, operation in enumerate(ordinary.functions[0].blocks[0].operations):
            assert operation.callee is not None
            if index < 2 * len(rotations) - 1 and index % 2 == 0:
                child = rotations[index // 2]
            elif index < 2 * len(rotations) - 1:
                child = double[1] if remaining else mixed
                remaining -= int(remaining > 0)
            else:
                child = positive[1]
            controlled_calls[operation.callee] = child
        controlled = controlled_gqsp_component_program(ordinary, controlled_calls)
    junk = tuple(
        register.name for register in source.walk_operator.prepare.junk_registers
    )
    return (
        HamiltonianGQSPContext(ordinary, preparation, junk),
        HamiltonianGQSPContext(controlled, preparation, junk),
    )


def resolve_controlled(
    source: Any,
    pair: tuple[HamiltonianGQSPContext, HamiltonianGQSPContext],
    polarity: int,
    references: dict[str, ProgramRecord],
) -> ProgramRecord:
    from qualtran import CtrlSpec

    library = _library(source.controlled(ctrl_spec=CtrlSpec(cvs=polarity)), 0)
    time, precision = source.t, source.precision
    walk_signature = tuple(_signature(source.walk_operator.signature))

    def owner(
        index: int,
    ) -> Callable[
        [FunctionRecord, ResolvedArgument, ResolvedArgument], HamiltonianGQSPContext
    ]:
        def context(
            walk: FunctionRecord, t: ResolvedArgument, eps: ResolvedArgument
        ) -> HamiltonianGQSPContext:
            assert walk.registers == walk_signature
            # Original transport can drop symbolic assumptions. Compare its
            # surviving expression without claiming a source object round trip.
            assert str(t) == str(time) and str(eps) == str(precision)
            return pair[index]

        return context

    factories: dict[str, ComponentFactory] = {}
    for key, program in references.items():

        def bind(
            name: str,
            registers: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
            child: ProgramRecord = program,
        ) -> ProgramRecord:
            assert registers is None or registers == child.functions[0].registers
            return _rename_program(child, name)

        factories[key] = bind
    factories.update(controlled_hamiltonian_gqsp_factories(owner(0), owner(1)))
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"cirq", "qualtran", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        return resolve_component_program(library, 0, factories)


def reference_programs(
    source: Any, context: WalkContext, preparation: ExactPair, selection: ProgramRecord
) -> dict[str, ProgramRecord]:
    walk = source.walk_operator
    return {
        type(child).__module__ + "." + type(child).__qualname__: program
        for child, program in (
            (walk, context.pair()[0]),
            (walk.block_encoding, context.block_pair[0]),
            (walk.prepare, preparation[0]),
            (walk.block_encoding.select, selection),
        )
    }


@pytest.fixture(scope="module")
def clean_case() -> tuple[
    Any,
    tuple[HamiltonianGQSPContext, HamiltonianGQSPContext],
    dict[str, ProgramRecord],
]:
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        HamiltonianSimulationByGQSP,
    )

    oracles = _oracles()
    with patch(
        "tests.unit.test_legacy_qubitization_walk_factories._oracles",
        return_value=oracles,
    ):
        walk, context, _ = context_fixture(False)
    source = HamiltonianSimulationByGQSP(walk, t=0.02, precision=0.01)
    return (
        source,
        contexts(source, context, oracles[4], oracles[2]),
        reference_programs(source, context, oracles[2], oracles[3][0]),
    )


@pytest.mark.parametrize("polarity", (0, 1))
def test_clean_controlled_hamiltonian_actual_transport_full_complex_all_histories(
    clean_case: tuple[
        Any,
        tuple[HamiltonianGQSPContext, HamiltonianGQSPContext],
        dict[str, ProgramRecord],
    ],
    polarity: int,
) -> None:
    from qualtran import CtrlSpec

    source, pair, references = clean_case
    native = resolve_controlled(source, pair, polarity, references)
    reference = source.controlled(ctrl_spec=CtrlSpec(cvs=polarity))
    pairs = compare_layers(
        native, reference, source_terminals(native) | scalar_terminals(native)
    )
    assert sum(type(child).__name__ == "Always" for child in pairs.values()) == 2
    bound, logical = bind_used_workspace(native)
    assert logical == 4
    expected = torch.as_tensor(reference.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=2e-10, rtol=0
    )
    assert_all_small_raw_histories(bound.circuit, logical, expected)
    initial = torch.arange(1, 17, dtype=torch.float64).to(torch.complex128)
    initial[1::2] *= 1j
    initial /= torch.linalg.vector_norm(initial)
    stride = 1 << (bound.circuit.n_wires - logical)
    state = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    state[::stride] = initial
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"cirq", "qualtran", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        circuit = Circuit(
            bound.circuit.n_wires, inputs=state, dtype=torch.complex128
        ).append(FieldOperation(bound.circuit.n_wires, bound.circuit.instructions))
        assert CircuitIR.from_json(circuit.to_ir().to_json()) == circuit.to_ir()
        dynamic = DynamicCircuit(
            bound.circuit.n_wires, inputs=state, dtype=torch.complex128
        )
        dynamic.append(
            FieldOperation(bound.circuit.n_wires, bound.circuit.instructions)
        )
        actual = run_dynamic(dynamic, shots=2, seed=17, strategy="trajectory")
        target = torch.zeros_like(state)
        target[::stride] = expected @ initial
        torch.testing.assert_close(
            actual.final_states.reshape(2, -1), target.expand(2, -1), atol=2e-10, rtol=0
        )


@pytest.fixture(scope="module", params=(False, True))
def hubbard_case(
    request: pytest.FixtureRequest,
) -> tuple[
    Any,
    tuple[HamiltonianGQSPContext, HamiltonianGQSPContext],
    bool,
    dict[str, ProgramRecord],
]:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        get_walk_operator_for_hubbard_model,
    )
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        HamiltonianSimulationByGQSP,
        _symbolic_hamsim_by_gqsp,
    )

    from flagquantum.compiler.registers.control.hubbard_select_component import (
        hubbard_select_program,
    )
    from flagquantum.compiler.registers.preparation.hubbard_prepare_component import (
        hubbard_prepare_program,
    )
    from flagquantum.ecosystem._hubbard_walk_component_library import (
        capture_hubbard_walk_context,
    )
    from flagquantum.ecosystem._packed_oracle_component_library import _prepare_angles

    symbolic = bool(request.param)
    source = (
        _symbolic_hamsim_by_gqsp.make()
        if symbolic
        else HamiltonianSimulationByGQSP(
            get_walk_operator_for_hubbard_model(2, 2, 1, 1), t=0.002, precision=0.01
        )
    )
    context = capture_hubbard_walk_context(source.walk_operator)
    prepare = source.walk_operator.prepare
    angles = _prepare_angles(2, prepare.t, prepare.u)
    assert angles is not None
    preparation = (
        hubbard_prepare_program(2, *angles),
        hubbard_prepare_program(2, *angles, adjoint=True),
    )
    pair = contexts(
        source,
        context,
        (
            hubbard_select_program(2, controlled=True),
            hubbard_select_program(2, controlled=True, adjoint=True),
        ),
        preparation,
    )
    return (
        source,
        pair,
        symbolic,
        reference_programs(source, context, preparation, hubbard_select_program(2)),
    )


@pytest.mark.parametrize("polarity", (0, 1))
def test_actual_hubbard_controlled_hamiltonian_available_graphs_and_symbolic_refusal(
    hubbard_case: tuple[
        Any,
        tuple[HamiltonianGQSPContext, HamiltonianGQSPContext],
        bool,
        dict[str, ProgramRecord],
    ],
    polarity: int,
) -> None:
    from qualtran import CtrlSpec

    source, pair, symbolic, references = hubbard_case
    record = resolve_controlled(source, pair, polarity, references)
    original = source.controlled(ctrl_spec=CtrlSpec(cvs=polarity))
    pairs = compare_layers(
        record, original, source_terminals(record) | scalar_terminals(record)
    )
    assert sum(type(child).__name__ == "Always" for child in pairs.values()) == 2
    if symbolic:
        # The only missing reachable body is the source-refused symbolic GQSP.
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_used_workspace(record)
    else:
        assert len(pairs) >= 348
        # Available graph equality never certifies the original dirty-Free
        # Hubbard domain. The clean fixture supplies the numerical certificate.
