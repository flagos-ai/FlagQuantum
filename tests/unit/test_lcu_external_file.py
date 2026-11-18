"""Whole original SELECT/LCU source captures and clean-history evidence."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path

import pytest

from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._component_snapshot import _snapshot
from tests.unit.measurement_certificate import certify_erasure_histories

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_lcu_source_files() -> None:
    import kickmix
    from qualtran.bloqs.block_encoding import lcu_block_encoding

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(lcu_block_encoding.__file__).parent
    for name, blob in (
        ("lcu_block_encoding.py", "679ac7964baf62288713078a5ec4dbce674cbece"),
        ("lcu_block_encoding_test.py", "0632799611bbd5ec2b7ca724533d4bbd1bbb7169"),
        ("lcu_block_encoding.ipynb", "4f457c91bedc3baaf94f7515242a67fffc4bf4b7"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("mode", ("select", "lcu", "lcu_zero", "lcu_one"))
def test_actual_original_named_and_packed_roots_all_measurement_histories(
    packed: bool, adjoint: bool, mode: str
) -> None:
    from attrs import evolve
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        _black_box_lcu_block,
        _black_box_select_block,
        _lcu_block,
        _select_block,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    if mode == "select":
        source = (_black_box_select_block if packed else _select_block)()
    else:
        source = (_black_box_lcu_block if packed else _lcu_block)()
        if mode != "lcu":
            source = evolve(source, control_val=int(mode == "lcu_one"))
    ports = tuple((reg.name, int(reg.total_bits())) for reg in source.signature)
    if adjoint:
        source = source.adjoint()
    ir, swaps = _snapshot(source, ports)
    logical = sum(width for _, width in ports)
    work = tuple(range(logical, ir.n_wires))
    assert swaps == 0
    assert logical == (30 if mode == "select" else 32 if mode == "lcu" else 33)
    restored = CircuitIR.from_json(ir.to_json())
    assert restored.to_json() == ir.to_json()
    count = certify_erasure_histories(
        restored.instructions, n_wires=ir.n_wires, zero_wires=work
    )
    assert count == sum(g.name == "measure" for g in ir.instructions)
    if mode == "select":
        assert count == 40 and len(work) == 5
        declaration = get_cost_value(source, QECGatesCost())
        assert count == declaration.measurement
        assert sum(g.name in {"t", "tdg"} for g in ir.instructions) == 230
        assert 4 * declaration.and_bloq + 7 * declaration.cswap == 230
    else:
        controlled = mode != "lcu"
        assert count == (89 if controlled else 86)
        assert (
            len(work) == 8
        )  # Complete exported-frame liveness reserves two extra names.
        assert (
            len(ir.instructions)
            == {
                "lcu": 2526,
                "lcu_zero": 2578,
                "lcu_one": 2576,
            }[mode]
        )
        assert sum(g.name in {"t", "tdg"} for g in ir.instructions) == (
            552 if controlled else 540
        )
    repeated, _ = _snapshot(source, ports)
    assert repeated.to_json() == ir.to_json()
