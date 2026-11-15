"""Check first-quantized SELECT and PREPARE models against the original cost graph."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.first_quantization import first_quantization_signature
from flagquantum.algorithms.first_quantization_kinetic import (
    prepare_kinetic_coordinate_on,
    prepare_power_two_signature,
    prepare_t_signature,
)
from flagquantum.algorithms.first_quantization_nu import (
    prepare_nu_m_on,
    prepare_nu_signature,
)
from flagquantum.algorithms.first_quantization_potential import (
    prepare_uv_signature,
    prepare_zeta_signature,
)
from flagquantum.algorithms.first_quantization_resource_model import (
    source_apply_nuclear_phase_cost,
    source_prepare_first_quantization_cost,
    source_prepare_ij_cost,
    source_prepare_ij_signature,
    source_prepare_power_two_cost,
    source_prepare_t_cost,
    source_prepare_tuv_cost,
    source_prepare_uv_cost,
    source_prepare_zeta_cost,
    source_select_first_quantization_cost,
    source_select_t_cost,
    source_select_uv_cost,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

pytestmark = pytest.mark.unit


def assert_first_quantization_signature(native, source):
    from qualtran import BQUInt, QAny, QBit

    def source_dtype(port):
        if port.dtype.encoding == "uint":
            return BQUInt(port.bitsize, dict(port.dtype.parameters)["bound"])
        if port.dtype.encoding == "bit":
            return QBit()
        assert port.dtype.encoding == "opaque"
        return QAny(port.bitsize)

    assert [
        (port.name, port.shape, port.side.value, source_dtype(port)) for port in native
    ] == [(port.name, port.shape, port.side.value, port.dtype) for port in source]


def assert_source_model(component: ComponentCost, source: object) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    report = estimate_ftqc_resources(component)
    counts = {k: v for k, v in asdict(report.logical.gates).items() if v}
    assert counts == get_cost_value(source, QECGatesCost()).asdict()
    assert report.logical.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert report.emitted is None


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("p", [1, 2, 6])
def test_kinetic_source_cost(p: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_t import (
        SelectTFirstQuantization,
    )

    source = SelectTFirstQuantization(p, 3)
    assert_source_model(source_select_t_cost(p), source)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(("p", "nuc"), [(1, 2), (2, 1), (4, 4), (4, 8)])
def test_nuclear_phase_source_cost(p: int, nuc: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import (
        ApplyNuclearPhase,
    )

    assert_source_model(
        source_apply_nuclear_phase_cost(p, nuc), ApplyNuclearPhase(p, nuc)
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("p", "atoms", "nuc"), [(2, 1, 1), (2, 2, 16), (4, 5, 4), (2, 2, 100)]
)
def test_potential_source_cost(p: int, atoms: int, nuc: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import (
        SelectUVFirstQuantization,
    )

    source = SelectUVFirstQuantization(p, 3, atoms, nuc)
    assert_source_model(source_select_uv_cost(p, atoms, nuc), source)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("p", "eta", "atoms", "m", "nuc"),
    [
        (2, 3, 2, 256, 16),
        (2, 5, 1, 1, 1),
        (3, 9, 3, 17, 1),
        (6, 10, 10, 256, 16),
        (2, 2, 2, 1, 100),
    ],
)
def test_select_graph_and_peak_width(
    p: int, eta: int, atoms: int, m: int, nuc: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import LeftDangle
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        SelectFirstQuantization,
    )

    source = SelectFirstQuantization(p, eta, atoms, 1, m_param=m, num_bits_nuc_pos=nuc)
    model = source_select_first_quantization_cost(
        p, eta, atoms, m_param=m, num_bits_nuc_pos=nuc
    )
    assert_source_model(model, source)
    assert_first_quantization_signature(
        first_quantization_signature(p, eta, atoms, m, prepare=False), source.signature
    )
    assert model.qubits == source.signature.n_qubits()

    graph = source.decompose_bloq()
    live = {edge for edge in graph.connections if edge.left.binst == LeftDangle}
    stages = []
    for node, incoming, outgoing in graph.iter_bloqnections():
        live.difference_update(incoming)
        if not isinstance(node.bloq, (Allocate, Free)):
            stages.append((str(node.bloq), sum(edge.shape for edge in live)))
        live.update(outgoing)
    assert stages == [
        (call.component.name, call.bystander_qubits) for call in model.calls
    ]


@pytest.mark.ftqc_reference
def test_source_one_bit_momentum_width_failure_is_explicit() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        SelectFirstQuantization,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = SelectFirstQuantization(1, 2, 2, 2)
    assert get_cost_value(source, QECGatesCost()).toffoli == 101
    with pytest.raises(RuntimeError, match="bitsize must be > 1"):
        get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="at least two momentum bits"):
        source_select_first_quantization_cost(1, 2, 2)


@pytest.mark.parametrize(("p", "eta", "precision"), [(1, 2, 1), (2, 3, 5), (6, 10, 8)])
@pytest.mark.ftqc_reference
def test_prepare_t_cost_paths(p: int, eta: int, precision: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, LeftDangle, RightDangle
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t import (
        PreparePowerTwoState,
        PrepareTFirstQuantization,
    )
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    power = PreparePowerTwoState(p)
    assert [
        (reg.name, reg.bitsize, reg.shape, reg.side.value)
        for reg in prepare_power_two_signature(p)
    ] == [(reg.name, reg.bitsize, reg.shape, reg.side.value) for reg in power.signature]
    assert_source_model(source_prepare_power_two_cost(p), power)
    with pytest.raises(DecomposeNotImplementedError):
        power.decompose_bloq()

    source = PrepareTFirstQuantization(p, eta, precision)
    assert [
        (reg.name, reg.bitsize, reg.shape, reg.side.value)
        for reg in prepare_t_signature(p)
    ] == [
        (reg.name, reg.bitsize, reg.shape, reg.side.value) for reg in source.signature
    ]
    declared = source_prepare_t_cost(p)
    assert_source_model(declared, source)
    assert source.bloq_counts() == {Toffoli(): 13, power: 2}
    assert [(call.component.name, call.repetitions) for call in declared.calls] == [
        ("Toffoli", 13),
        ("PreparePowerTwoState", 2),
    ]

    for adjoint in (False, True):
        directional = source.adjoint() if adjoint else source
        composite = directional.decompose_bloq()
        assert_source_model(source_prepare_t_cost(p, adjoint=adjoint), directional)
        assert_source_model(
            source_prepare_t_cost(p, decomposed=True, adjoint=adjoint), composite
        )
        # These calls are independent. Compare the port map in either schedule.
        assert {
            (node.bloq, incoming[0].left.reg.name, outgoing[0].right.reg.name)
            for node, incoming, outgoing in composite.iter_bloqnections()
            if incoming[0].left.binst == LeftDangle
            and outgoing[0].right.binst == RightDangle
        } == {
            (child.adjoint() if adjoint else child, port, port)
            for child, port in (
                (PrepareUniformSuperposition(3), "w"),
                (power, "r"),
                (power, "s"),
            )
        }


@pytest.mark.parametrize(("eta", "precision"), [(1, 8), (2, 1), (10, 8)])
@pytest.mark.ftqc_reference
def test_prepare_ij_cost(eta: int, precision: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization import (
        prepare,
        select_and_prepare,
    )

    for source_module in (prepare, select_and_prepare):
        source = source_module.UniformSuperpostionIJFirstQuantization(eta, precision)
        signature = source_prepare_ij_signature(eta)
        assert [
            (reg.name, reg.bitsize, reg.shape, reg.side.value) for reg in signature
        ] == [
            (reg.name, reg.bitsize, reg.shape, reg.side.value)
            for reg in source.signature
        ]
        assert_source_model(source_prepare_ij_cost(eta, precision), source.adjoint())
        assert_source_model(source_prepare_ij_cost(eta, precision), source)
        with pytest.raises(DecomposeNotImplementedError):
            source.decompose_bloq()


@pytest.mark.parametrize("adjoint", [False, True])
def test_kinetic_prepare_resource_report_separates_emitted_coordinate(
    adjoint: bool,
) -> None:
    from flagquantum.compiler.registers.preparation.uniform_preparation import (
        uniform_preparation_ir,
        uniform_preparation_requirements,
    )
    from flagquantum.dynamic import DynamicCircuit

    target, work, slots = uniform_preparation_requirements(3)
    wires = tuple(reversed(range(target)))
    workspace = tuple(range(target, target + work))
    classical_bits = tuple(range(slots))
    coordinate = uniform_preparation_ir(
        3, wires, workspace, classical_bits, adjoint=adjoint
    )
    circuit = DynamicCircuit(target + work).append(
        prepare_kinetic_coordinate_on(
            wires, workspace=workspace, classical_bits=classical_bits, adjoint=adjoint
        )
    )
    ir = circuit.to_ir()
    assert ir.instructions == coordinate
    declared = estimate_ftqc_resources(source_prepare_t_cost(6))
    composite = estimate_ftqc_resources(source_prepare_t_cost(6, decomposed=True))
    emitted = estimate_ftqc_resources(
        source_prepare_t_cost(6, decomposed=True).calls[0].component,
        emitted_ir=ir,
    )
    assert declared.logical.gates.toffoli == 21
    assert composite.logical.gates.toffoli == 8
    assert (
        declared.logical.peak_logical_qubits
        == composite.logical.peak_logical_qubits
        == 18
    )
    assert declared.emitted is composite.emitted is None
    assert emitted.emitted is not None
    assert emitted.emitted.t_count == 20
    assert emitted.emitted.unconditional["measure"] == 5
    assert emitted.emitted.declared_qubits == 6
    ij = estimate_ftqc_resources(source_prepare_ij_cost(10))
    assert 2 * (ij.logical.gates.toffoli + declared.logical.gates.toffoli) == 126


@pytest.mark.parametrize(
    ("atoms", "charge", "precision"),
    [(1, 0, 1), (1, 1, 8), (2, 4, 8), (9, 10, 16), (33, 15, 2), (2, 2**54 + 1, 8)],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_prepare_zeta_cost(
    atoms: int, charge: int, precision: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, QAny, Side
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta import (
        PrepareZetaState,
    )

    source = PrepareZetaState(atoms, charge, precision)
    if adjoint:
        source = source.adjoint()
    index = source.signature.get_left("l")
    assert index.side == Side.THRU
    assert index.bitsize == (atoms - 1).bit_length()
    (native,) = prepare_zeta_signature(atoms)
    assert native.name == index.name == "l"
    assert native.side.value == index.side.value
    assert native.shape == index.shape
    assert native.dtype.encoding == "opaque"
    assert QAny(native.bitsize) == index.dtype
    assert_source_model(
        source_prepare_zeta_cost(atoms, charge, adjoint=adjoint), source
    )
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.parametrize(
    ("p", "eta", "atoms", "m", "charge", "precision"),
    [
        (1, 2, 1, 3, 0, 1),
        (2, 4, 1, 2, 1, 16),
        (2, 3, 2, 5, 4, 8),
        (3, 4, 33, 9, 15, 2),
        (6, 10, 10, 32768, 10, 8),
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_prepare_uv_source_graph(
    p: int, eta: int, atoms: int, m: int, charge: int, precision: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import LeftDangle, QAny, QBit, RightDangle
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv import (
        PrepareUVFirstQuantization,
    )

    source = PrepareUVFirstQuantization(p, eta, atoms, m, charge, precision)
    if adjoint:
        source = source.adjoint()
    signature = prepare_uv_signature(p, atoms, m)
    assert [
        (
            reg.name,
            reg.shape,
            reg.side.value,
            QBit() if reg.dtype.encoding == "bit" else QAny(reg.bitsize),
        )
        for reg in signature
    ] == [(reg.name, reg.shape, reg.side.value, reg.dtype) for reg in source.signature]
    model = source_prepare_uv_cost(p, atoms, m, charge, adjoint=adjoint)
    assert_source_model(model, source)
    assert model.qubits == source.signature.n_qubits()
    composite = source.decompose_bloq()
    assert_source_model(model, composite)
    live = {edge for edge in composite.connections if edge.left.binst == LeftDangle}
    stages = []
    for node, incoming, outgoing in composite.iter_bloqnections():
        live.difference_update(incoming)
        assert all(edge.left.binst == LeftDangle for edge in incoming)
        assert all(edge.right.binst == RightDangle for edge in outgoing)
        child = (
            prepare_zeta_signature(atoms)
            if str(node.bloq) == "PrepareZetaState"
            else prepare_nu_signature(p, m)
        )
        expected = {(reg.name, idx) for reg in child for idx in reg.all_idxs()}
        for connections in (incoming, outgoing):
            assert {
                (edge.left.reg.name, edge.left.idx) for edge in connections
            } == expected
            assert all(
                (edge.left.reg.name, edge.left.idx)
                == (edge.right.reg.name, edge.right.idx)
                for edge in connections
            )
        stages.append((str(node.bloq), sum(edge.shape for edge in live)))
        live.update(outgoing)
    assert stages == [
        (call.component.name, call.bystander_qubits) for call in model.calls
    ]


@pytest.mark.parametrize("adjoint", [False, True])
def test_prepare_uv_m_child_uses_only_the_bound_word_and_workspace(
    adjoint: bool,
) -> None:
    from flagquantum.compiler.registers.layout.partition import partition_register
    from flagquantum.compiler.registers.preparation.uniform_preparation import (
        uniform_preparation_requirements,
    )
    from flagquantum.dynamic import DynamicCircuit

    signature = prepare_uv_signature(2, 2, 3)
    layout = tuple((reg.name, int(reg.total_bits())) for reg in signature)
    width = int(signature.n_qubits())
    ports = partition_register(tuple(range(width)), layout)
    _, work, slots = uniform_preparation_requirements(3)
    workspace = tuple(range(width, width + work))
    circuit = DynamicCircuit(width + work).append(
        prepare_nu_m_on(
            3,
            ports["m"],
            workspace=workspace,
            classical_bits=tuple(range(slots)),
            adjoint=adjoint,
        )
    )
    stream = circuit.to_ir().instructions
    assert {wire for gate in stream for wire in gate.wires} == set(
        (*ports["m"], *workspace)
    )
    assert sum(gate.name == "measure" for gate in stream) == 5


def test_prepare_uv_roundtrip_resources() -> None:
    forward = estimate_ftqc_resources(source_prepare_uv_cost(6, 10, 32768, 10))
    reverse = estimate_ftqc_resources(
        source_prepare_uv_cost(6, 10, 32768, 10, adjoint=True)
    )
    assert (
        forward.logical.gates.total_t_count() + reverse.logical.gates.total_t_count()
        == 2516
    )
    assert forward.logical.peak_logical_qubits == 77
    assert reverse.logical.peak_logical_qubits == 50
    assert forward.emitted is reverse.emitted is None


@pytest.mark.parametrize(
    ("bits", "eta", "charge", "precision"),
    [(1, 1, 0, 1), (8, 3, 0, 5), (16, 10, 10, 8), (16, 4, 2**54, 8)],
)
@pytest.mark.ftqc_reference
def test_prepare_tuv_source_cost(
    bits: int, eta: int, charge: int, precision: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        PrepareTUVSuperpositions,
    )

    source = PrepareTUVSuperpositions(bits, eta, charge, precision)
    model = source_prepare_tuv_cost(bits, eta, charge, precision)
    assert [(reg.name, reg.bitsize) for reg in source.signature.lefts()] == [
        ("tuv", 1),
        ("uv", 1),
    ]
    for operation in (source, source.adjoint()):
        assert_source_model(model, operation)
        with pytest.raises(DecomposeNotImplementedError):
            operation.decompose_bloq()


@pytest.mark.parametrize(
    ("p", "eta", "atoms", "charge", "m", "term_bits", "rotation_bits"),
    [
        (1, 2, 1, 0, 3, 1, 1),
        (2, 3, 1, 1, 5, 9, 2),
        (2, 2, 4, 4, 64, 7, 3),
        (3, 9, 5, 15, 9, 12, 8),
        (6, 10, 10, 10, 32768, 16, 8),
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_prepare_first_quantization_source_graph(
    p: int,
    eta: int,
    atoms: int,
    charge: int,
    m: int,
    term_bits: int,
    rotation_bits: int,
    adjoint: bool,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import LeftDangle, RightDangle
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        PrepareFirstQuantization,
    )

    source = PrepareFirstQuantization(
        p,
        eta,
        atoms,
        charge,
        m_param=m,
        num_bits_t=term_bits,
        num_bits_rot_aa=rotation_bits,
    )
    if adjoint:
        source = source.adjoint()
    model = source_prepare_first_quantization_cost(
        p,
        eta,
        atoms,
        charge,
        m_param=m,
        num_bits_t=term_bits,
        num_bits_rot_aa=rotation_bits,
        adjoint=adjoint,
    )
    assert_source_model(model, source)
    signature = first_quantization_signature(p, eta, atoms, m, prepare=True)
    assert_first_quantization_signature(
        signature.adjoint() if adjoint else signature, source.signature
    )
    assert model.qubits == source.signature.n_qubits()
    graph = source.decompose_bloq()
    live = {edge for edge in graph.connections if edge.left.binst == LeftDangle}
    assert [
        (edge.left.reg.name, edge.right.reg.name)
        for edge in live
        if edge.right.binst == RightDangle
    ] == [("plus_t", "plus_t")]
    stages = []
    for node, incoming, outgoing in graph.iter_bloqnections():
        live.difference_update(incoming)
        stages.append((str(node.bloq), sum(edge.shape for edge in live)))
        if str(node.bloq).startswith("PrepareUV"):
            assert {
                (edge.left.reg.name, edge.right.idx)
                for edge in incoming
                if edge.right.reg.name == "nu"
            } == {("nu_x", (0,)), ("nu_y", (1,)), ("nu_z", (2,))}
            assert {
                (edge.right.reg.name, edge.left.idx)
                for edge in outgoing
                if edge.left.reg.name == "nu"
            } == {("nu_x", (0,)), ("nu_y", (1,)), ("nu_z", (2,))}
            assert any(
                edge.left.reg.name == "succ_nu" and edge.right.reg.name == "flag_nu"
                for edge in incoming
            )
        live.update(outgoing)
    assert stages == [
        (call.component.name, call.bystander_qubits) for call in model.calls
    ]


@pytest.mark.parametrize(
    ("eta", "m", "missing", "error"),
    [(1, 8, "i.*j|j.*i", "at least two particles"), (2, 1, "target", "m_param >= 2")],
)
@pytest.mark.ftqc_reference
def test_prepare_first_quantization_source_zero_width_failure(
    eta: int, m: int, missing: str, error: str
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqError
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        PrepareFirstQuantization,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value

    with pytest.raises((BloqError, RuntimeError), match=missing):
        get_cost_value(PrepareFirstQuantization(2, eta, 1, 1, m_param=m), QubitCount())
    with pytest.raises(ValueError, match=error):
        source_prepare_first_quantization_cost(2, eta, 1, 1, m_param=m)
