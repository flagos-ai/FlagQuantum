"""Check locked component identities, bank layouts and copy multiplicities."""

from typing import Any

import pytest

from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.lookup.select_swap_qrom_calls import (
    SelectSwapCall,
    SelectSwapRouteCall,
    SelectSwapXorCall,
    select_swap_qrom_calls,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_call_key(child: Any) -> SelectSwapCall:
    from qualtran.bloqs.arithmetic.bitwise import Xor
    from qualtran.bloqs.data_loading import QROM
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero

    if isinstance(child, QROM):
        return QromTable(
            child.data_shape,
            child.selection_bitsizes,
            child.target_bitsizes,
            child.target_shapes,
            child.num_controls,
            (
                tuple(tuple(map(int, data.flat)) for data in child.data)
                if child.has_data()
                else None
            ),
        )
    if isinstance(child, SwapWithZero):
        return SelectSwapRouteCall(
            child.target_bitsize,
            child.selection_bitsizes,
            child.n_target_registers,
            child.uncompute,
        )
    assert isinstance(child, Xor)
    return SelectSwapXorCall(child.dtype.num_qubits)


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "default_line",
        "default_grid",
        "blocked",
        "fully_blocked",
        "mixed",
        "symbolic",
        "symbolic_default",
    ],
)
def test_select_swap_direct_calls(kind: str, dirty: bool) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.data_loading import SelectSwapQROM
    from qualtran.resource_counting.generalizers import ignore_split_join
    from sympy import symbols

    logs: tuple[SymbolicInt, ...] | None = None
    if kind.startswith("symbolic"):
        n, m, width, row_log, column_log, control = symbols(
            "N M b k1 k2 c", positive=True, integer=True
        )
        shape = (n, m) if kind == "symbolic" else (n,)
        logs = (row_log, column_log) if kind == "symbolic" else None
        table = QromTable.from_shape(shape, (width, width), num_controls=control)
        source = SelectSwapQROM.build_from_bitsize(
            shape,
            (width, width),
            log_block_sizes=logs,
            num_controls=control,
            use_dirty_ancilla=dirty,
        )
    else:
        data = np.arange(5) if kind == "default_line" else np.arange(25).reshape(5, 5)
        if kind == "fully_blocked":
            data = np.arange(4)
        logs = {"blocked": (1, 1), "fully_blocked": (2,), "mixed": (1, 0)}.get(kind)
        table = QromTable.from_data(data, data + 1)
        source = SelectSwapQROM.build_from_data(
            data,
            data + 1,
            log_block_sizes=logs,
            use_dirty_ancilla=dirty,
        )
    graph, _ = source.call_graph(max_depth=1, generalizer=ignore_split_join)
    expected = {
        source_call_key(child): data["n"] for _, child, data in graph.edges(data=True)
    }
    assert select_swap_qrom_calls(table, logs, dirty=dirty) == expected
