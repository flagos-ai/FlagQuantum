# SPDX-License-Identifier: Apache-2.0
"""Replace foreign binding storage with existing native consumer paths."""

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import cast

import pytest

from flagquantum.compiler.artifacts.operand_array import OperandArray
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.registers.primitives.operand_binding import (
    bind_operand,
    identify_operand,
)
from flagquantum.ecosystem._physical_basis_session import PhysicalBasisSession
from tools.development.docstrings import get_markdown_docstring_lines

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "python_binding_scaffolding.json").read_text()
)


@pytest.mark.ftqc_reference
def test_complete_scaffolding_and_direct_dependency_blobs() -> None:
    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    pending = [
        "kickmix/src/kickmix/py/" + name
        for name in ("types.pybind.h", "util.pybind.cc", "util.pybind.h")
    ]
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        expected = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    verified = set()
    while pending:
        path = pending.pop()
        if path in verified:
            continue
        raw = (reference / path).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            == expected[path]
        )
        verified.add(path)
        for name in re.findall(r'^#include "([^"]+)"', raw.decode(), re.M):
            local = Path(path).parent / name
            pending.append(
                str(local) if (reference / local).exists() else "kickmix/src/" + name
            )
    assert len(verified) == 16


@pytest.mark.parametrize("case", _DATA)
def test_all_six_foreign_storage_families_use_native_owners(
    case: dict[str, object],
) -> None:
    operands = [
        OperandIdentity(OperandKind.QUANTUM, 5),
        OperandIdentity(OperandKind.CLASSICAL, 3),
        OperandIdentity(OperandKind.X_CLASSICAL, 2),
        OperandIdentity(OperandKind.X_BOOLEAN, 0),
        OperandIdentity(OperandKind.X_BOOLEAN, 1),
    ]
    assert [identify_operand(bind_operand(item)) for item in operands] == operands
    array = OperandArray(operands)

    def source_text(item: OperandIdentity) -> str:
        if item.kind is OperandKind.X_BOOLEAN:
            return f"xbool({bool(item)})"
        return item.diagnostic_text

    assert [source_text(item) for item in array] == case["array"]
    view = array[::-1]
    del array
    assert [source_text(item) for item in view] == case["reverse"]
    simulator = PhysicalBasisSession(cast(int, case["batch_size"]), debug_print=False)
    for operand in operands[:2]:
        simulator.write_across_shots(operand, case["input_mask"])
    assert [
        simulator.read_across_shots(item, out=int) for item in operands[:2]
    ] == case["scalar"]
    circuit = read_kmx_circuit('APPEND_TO_REGISTER q5 r0\nREGISTER r0 "data"')
    simulator.use_same_registers_as(circuit)
    assert (
        simulator.read_across_shots(OperandIdentity(OperandKind.REGISTER, 0), out=int)
        == case["register"]
    )
    assert simulator.read_across_shots("data", out=int) == case["name"]
    with pytest.raises(ValueError, match="register identity"):
        bind_operand(OperandIdentity(OperandKind.REGISTER, 0))
    with pytest.raises(ValueError, match="typed identities"):
        OperandArray(cast(list[OperandIdentity], [object()]))


@pytest.mark.parametrize(
    "raw,lines",
    [
        ("", []),
        ("\n    Alpha\n    Beta\n", ["Alpha", "Beta"]),
        ("\n\t A\n\t B", ["A", "B"]),
        ("text\n  rest", ["text", "rest"]),
        ('doc """ allowed', ['doc """ allowed']),
        ("x" * 80 + "\n", ["x" * 80]),
        ("x" * 81, ["x" * 81]),
        ("@signature " + "x" * 100, ["@signature " + "x" * 100]),
        ("@overload " + "x" * 100, ["@overload " + "x" * 100]),
        ("x" * 90 + "https://example.test", ["x" * 90 + "https://example.test"]),
    ],
)
def test_existing_documentation_consumer_retains_python_policy(
    raw: str,
    lines: list[str],
) -> None:
    consumer = type("Consumer", (), {"__doc__": raw})
    assert get_markdown_docstring_lines(consumer) == ["## `Consumer`", *lines]
