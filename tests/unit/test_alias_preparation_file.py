"""Original coherent alias preparation tutorial fixtures and probability grid."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from sympy import ceiling, simplify, symbols

from examples.algorithms.alias_preparation_catalog import (
    AliasPreparationExample,
    dense_alias_example,
    sparse_alias_example,
    symbolic_alias_example,
)
from flagquantum.algorithms.primitives.alias_sampling import (
    alias_sampling_table,
    sparse_alias_sampling_table_from_dense,
)
from flagquantum.algorithms.primitives.symbolic_alias_preparation import (
    AliasPreparationShape,
)
from flagquantum.circuit import Circuit
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.state_preparation import state_preparation_alias_sampling

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    base = Path(state_preparation_alias_sampling.__file__).parent
    for name, digest in (
        (
            "state_preparation_alias_sampling.ipynb",
            "081add39d9e5a13a0e215483e429b6f0d9e3ff84",
        ),
        (
            "state_preparation_alias_sampling.py",
            "4850ef95d9ebf58b0e062505a3482db2f5307443",
        ),
        (
            "state_preparation_alias_sampling_test.py",
            "c5141c0e4beadb9a890dc3fc035601252a6c0454",
        ),
    ):
        raw = (base / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return state_preparation_alias_sampling


def assert_signature(native: Signature, source: Any) -> None:
    assert [(r.name, r.side.name) for r in native] == [
        (r.name, r.side.name) for r in source.signature
    ]
    for actual, expected in zip(native, source.signature, strict=True):
        assert simplify(actual.bitsize - expected.bitsize) == 0
        assert actual.shape == expected.shape
        kind = type(expected.dtype).__name__
        assert (
            actual.dtype.encoding
            == {"BQUInt": "uint", "QBit": "bit", "QAny": "opaque"}[kind]
        )
        if kind == "BQUInt":
            assert (
                simplify(
                    dict(actual.dtype.parameters)["bound"]
                    - expected.dtype.iteration_length
                )
                == 0
            )


@pytest.mark.parametrize("factory", ("dense", "sparse", "from_dense"))
def test_concrete_alias_factory(source_module: Any, factory: str) -> None:
    if factory == "dense":
        example = dense_alias_example()
        source = source_module._state_prep_alias()
    else:
        from_dense = factory == "from_dense"
        example = sparse_alias_example(from_dense=from_dense)
        source = (
            source_module._sparse_state_prep_alias_from_list()
            if from_dense
            else source_module._sparse_state_prep_alias()
        )
    assert_signature(example.signature, source)
    table = example.table
    assert table.mu == source.mu == 3
    assert table.normalization == source.l1_norm_of_coeffs
    assert table.alternates == tuple(source.alt)
    assert table.keep == tuple(source.keep)
    if table.indices is not None:
        assert table.indices == tuple(source.index)
    assert_source_model(example.resource_model(), source)
    assert_source_model(example.resource_model(), source.adjoint())


@pytest.mark.parametrize("sparse", (False, True))
def test_symbolic_alias_factory(source_module: Any, sparse: bool) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    plan = symbolic_alias_example(sparse=sparse)
    source = (
        source_module._sparse_state_prep_alias_symb()
        if sparse
        else source_module._state_prep_alias_symb()
    )
    assert_signature(plan.signature, source)
    assert plan.l1_norm_of_coeffs == source.l1_norm_of_coeffs
    for legacy in (False, True):
        model = plan.resource_model(legacy_shims=legacy)
        for operation in (source, source.adjoint()):
            gates = get_cost_value(operation, QECGatesCost(legacy_shims=legacy))
            assert all(
                simplify(getattr(model, key) - getattr(gates, key)) == 0
                for key in (
                    "and_bloq",
                    "clifford",
                    "measurement",
                    "rotation",
                    "cswap",
                    "t",
                    "toffoli",
                )
            )
            assert (
                simplify(model.logical_qubits - get_cost_value(operation, QubitCount()))
                == 0
            )


def test_symbolic_precision_recovers_integral_mu(source_module: Any) -> None:
    rows, total, mu = symbols(r"L \lambda \mu", integer=True)
    precision = 2 ** (-mu) * total / rows
    plan = AliasPreparationShape.from_precision(rows, total, precision)
    source = source_module.StatePreparationAliasSampling.from_n_coeff(
        rows, total, precision=precision
    )
    assert simplify(plan.probability_bits - mu) == simplify(source.mu - mu) == 0


@pytest.mark.parametrize(
    "sites,epsilon", ((2, 0.01), (2, 0.003), (3, 0.003), (4, 0.005), (7, 0.008))
)
def test_original_ising_probability_grid(
    source_module: Any, sites: int, epsilon: float
) -> None:
    from qualtran.bloqs.chemistry.ising import get_1d_ising_lcu_coeffs

    weights = tuple(map(float, get_1d_ising_lcu_coeffs(sites)))
    assert_alias_probabilities(weights, epsilon)


@pytest.mark.parametrize(
    "weights,sparse",
    (
        ((1 / 8,) * 8, False),
        ((1 / 8,) * 8 + (0,) * 8, True),
        ((1,) * 6 + (0,) * 4, True),
    ),
)
def test_original_uniform_and_sparse_probabilities(
    source_module: Any, weights: tuple[float, ...], sparse: bool
) -> None:
    assert_alias_probabilities(weights, 0.2, sparse=sparse)


def assert_alias_probabilities(
    weights: tuple[float, ...], epsilon: float, *, sparse: bool = False
) -> None:
    total = sum(weights)
    factory = sparse_alias_sampling_table_from_dense if sparse else alias_sampling_table
    table = factory(weights, precision=epsilon * total)
    example = AliasPreparationExample(table)
    circuit = Circuit(example.qubit_count)
    circuit.append(example)
    result = execute_sparse_register(circuit.to_ir(), {0: 1}, seed=27)
    width = (len(weights) - 1).bit_length()
    probabilities = [0.0] * (1 << width)
    for index, value in result.items():
        probabilities[index >> (example.qubit_count - width)] += abs(value) ** 2
    assert sum(probabilities) == pytest.approx(1, abs=3e-12)
    assert probabilities[: len(weights)] == pytest.approx(
        [weight / total for weight in weights], abs=epsilon
    )
    assert probabilities[len(weights) :] == pytest.approx(
        [0] * ((1 << width) - len(weights)), abs=3e-12
    )


def test_original_symbolic_t_complexity(source_module: Any) -> None:
    from qualtran.symbolics import log2

    rows, support, total, epsilon = symbols(
        r"L d \lambda \epsilon", positive=True, integer=True
    )
    dense = AliasPreparationShape.from_precision(rows, total, epsilon)
    sparse = AliasPreparationShape.from_sparse_precision(rows, support, total, epsilon)
    dense_cost = dense.resource_model(legacy_shims=True)
    sparse_cost = sparse.resource_model(legacy_shims=True)
    log_rows, log_support = ceiling(log2(rows)), ceiling(log2(support))
    mu = sparse.probability_bits
    dense_t = 4 * rows + 8 * dense.probability_bits + 19 * log_rows - 8
    sparse_t = 4 * support + 8 * mu + 7 * log_rows + 12 * log_support - 8
    sparse_clifford = (
        support * mu * log_rows**2
        + 13 * support
        + 47 * mu
        + 10 * log_rows
        + 52 * log_support
        - 12
    )
    for actual, expected in ((dense_cost, dense_t), (sparse_cost, sparse_t)):
        assert (
            simplify(actual.t + 4 * actual.and_bloq + 7 * actual.cswap - expected) == 0
        )
    # The original TComplexity includes the Clifford cost of expanding And and CSwap.
    expanded_clifford = sparse_cost.clifford + 10 * (
        sparse_cost.and_bloq + sparse_cost.cswap
    )
    assert simplify(expanded_clifford - sparse_clifford) == 0
    assert sparse_cost.rotation == 2


@pytest.mark.parametrize("sparse", (False, True))
def test_alias_singleton_declaration_keeps_selection_and_filters_empty_junk(
    sparse: bool, source_module: Any
) -> None:
    from flagquantum.algorithms.primitives.alias_preparation import (
        alias_black_box_prepare,
    )
    from flagquantum.algorithms.primitives.alias_sampling import (
        sparse_alias_sampling_table,
    )

    if sparse:
        source = source_module.SparseStatePreparationAliasSampling.from_sparse_dict(
            {2: 1.0}, 3, precision=0.25
        )
        table = sparse_alias_sampling_table({2: 1.0}, 3, precision=0.25)
    else:
        source = source_module.StatePreparationAliasSampling.from_probabilities(
            [1.0], precision=0.25
        )
        table = alias_sampling_table((1.0,), precision=0.25)
    declaration = AliasPreparationShape.from_table(table)
    assert_signature(declaration.signature, source)
    assert declaration.signature[0].name == "selection"
    assert declaration.signature[0].bitsize == (2 if sparse else 0)
    box = alias_black_box_prepare(table)
    assert box.junk_layout == tuple(
        (reg.name, reg.bitsize) for reg in declaration.junk_registers
    )
    assert box.normalization == declaration.l1_norm_of_coeffs
