"""Complete stored Montgomery file owners retain source channels and failures."""

from itertools import product
from math import sqrt

import pytest

from flagquantum.compiler.registers.modular.montgomery_multiplication_component import (
    montgomery_multiplication_program,
)
from flagquantum.compiler.registers.modular.windowed_multiplication_component import (
    windowed_multiplication_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_component_hierarchy_control import compare_layers

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("n,w,p", [(3, 1, 3), (4, 2, 5), (5, 5, 13), (6, 2, 7)])
@pytest.mark.parametrize("wrapper", [False, True])
def test_complete_source_forward_root_connections(
    n: int, w: int, p: int, wrapper: bool
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import (
        DirtyOutOfPlaceMontgomeryModMul,
        _DirtyOutOfPlaceMontgomeryModMulImpl,
    )

    factory = (
        montgomery_multiplication_program
        if wrapper
        else windowed_multiplication_program
    )
    source = (
        DirtyOutOfPlaceMontgomeryModMul(n, w, p)
        if wrapper
        else _DirtyOutOfPlaceMontgomeryModMulImpl(n, w, p)
    )
    graph = factory(n, w, p)
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})


@pytest.mark.parametrize(
    "n,w,p", [(3, 1, 3), (4, 2, 5), (5, 1, 9), (5, 5, 13), (6, 2, 7)]
)
def test_decomposed_source_full_complex_history_outputs_and_inverse(
    n: int, w: int, p: int
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import (
        DirtyOutOfPlaceMontgomeryModMul,
    )

    source = DirtyOutOfPlaceMontgomeryModMul(n, w, p)
    composite = source.decompose_bloq()
    inputs = (tuple(range(n)), tuple(range(n, 2 * n)))
    spectator = 2 * n
    bound = bind_register_program(
        montgomery_multiplication_program(n, w, p),
        inputs,
        clean=tuple(range(spectator + 1, spectator + 14 * n + 30)),
        allow_workspace_outputs=True,
    )
    total = max(q for gate in bound.circuit.instructions for q in gate.wires) + 1
    live = {q for word in bound.outputs for q in word}
    inverse = bind_register_program(
        montgomery_multiplication_program(n, w, p, adjoint=True),
        bound.outputs,
        clean=tuple(q for q in range(total) if q not in live and q != spectator),
        input_ownership=tuple((q, False) for word in bound.outputs[2:] for q in word),
    )
    assert inverse.outputs == inputs
    initial: dict[int, complex] = {}
    expected: dict[int, complex] = {}
    cases = list(product(range(p), range(p), (0, 1)))
    norm = sqrt(sum(abs(complex(i + 1, 2 - 3 * i)) ** 2 for i in range(len(cases))))

    def encode(words: tuple[tuple[int, ...], ...], values: tuple[int, ...]) -> int:
        return sum(
            ((value >> (len(word) - 1 - i)) & 1) << (total - q - 1)
            for word, value in zip(words, values, strict=True)
            for i, q in enumerate(word)
        )

    for i, (x, y, s) in enumerate(cases):
        observed = composite.on_classical_vals(x=x, y=y)
        values = tuple(
            int(observed[name])
            for name in ("x", "y", "target", "qrom_indices", "reduced")
        )
        if (n, w, p, x, y) == (5, 5, 13, 5, 9):
            assert (
                observed["target"] == 27
                and source.on_classical_vals(x=x, y=y)["target"] == 1
            )
        if (n, w, p, x, y) == (5, 1, 9, 3, 3):
            assert (
                observed["target"] == 18
                and source.on_classical_vals(x=x, y=y)["target"] == 0
            )
        amplitude = complex(i + 1, 2 - 3 * i) / norm
        mask = s << (total - spectator - 1)
        initial[encode(inputs, (x, y)) | mask] = amplitude
        expected[encode(bound.outputs, values) | mask] = amplitude
    for seed in (0, 7, 23):
        actual = execute_sparse_register(
            CircuitIR(total, bound.circuit.instructions), initial, seed=seed
        )
        assert (
            max(
                abs(actual.get(k, 0) - expected.get(k, 0))
                for k in actual.keys() | expected.keys()
            )
            < 3e-12
        )
        restored = execute_sparse_register(
            CircuitIR(total, inverse.circuit.instructions), actual, seed=seed
        )
        assert (
            max(
                abs(restored.get(k, 0) - initial.get(k, 0))
                for k in restored.keys() | initial.keys()
            )
            < 3e-12
        )


def test_complete_frozen_file_definition_ownership() -> None:
    import ast
    from pathlib import Path

    from qualtran.bloqs.mod_arithmetic import mod_multiplication

    audit = (
        Path(__file__).parents[2]
        / "docs/development/MODULAR_MULTIPLICATION_FILE_AUDIT.md"
    ).read_text()
    count = 0
    for node in ast.parse(Path(mod_multiplication.__file__).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            assert f"`{node.name}`" in audit
            count += 1
            for method in node.body if isinstance(node, ast.ClassDef) else ():
                if isinstance(method, ast.FunctionDef):
                    assert f"`{node.name}.{method.name}`" in audit
                    count += 1
    assert count == 45
