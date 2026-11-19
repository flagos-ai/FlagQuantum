"""Compare default component presentation with the unchanged base interface."""

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.testing.component_examples import ComponentDocumentation

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")


@dataclass(frozen=True)
class DisplayName:
    title: str

    def __str__(self) -> str:
        return self.title


@pytest.fixture(scope="module", autouse=True)
def frozen_default_interface() -> None:
    from qualtran._infra import bloq

    data = Path(bloq.__file__).read_bytes()
    assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == (
        "cef1f2f30bdc849048f50186616b4efce9877d8f"
    )


@pytest.mark.parametrize("title", ["Gate", "abcdefghij", "abcdefghijk", "Phase†Phase†"])
@pytest.mark.parametrize(
    "side", [RegisterSide.LEFT, RegisterSide.RIGHT, RegisterSide.THRU]
)
@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
def test_default_wire_title_and_all_directional_index_labels(
    title: str, side: RegisterSide, shape: tuple[int, ...]
) -> None:
    from qualtran import Bloq, QAny, Side
    from qualtran import Register as SourceRegister

    source_register = SourceRegister(
        "word", QAny(3), shape=shape, side=Side(side.value)
    )
    source = DisplayName(title)
    native_register = Register("word", RegisterType("opaque", 3), shape, side)
    indices = tuple(native_register.all_idxs())
    keys = tuple(f"wire{i}" for i in range(len(indices)))
    inputs = keys if side & RegisterSide.LEFT else ()
    outputs = keys if side & RegisterSide.RIGHT else ()
    step = RegisterStep(
        title,
        inputs,
        outputs,
        input_registers=(native_register,) if inputs else (),
        output_registers=(native_register,) if outputs else (),
    )
    score = layout_register_score(
        dict.fromkeys(keys, 3), inputs, (step,), outputs=outputs
    )
    actual = register_score_data(score)
    assert (
        actual["vlines"][0]["label"]
        == Bloq.wire_symbol(source, None).json_dict()["symb_attributes"]
    )
    ports = [port for port in actual["soqs"] if not port["ident"].startswith("dang,")]
    assert len(ports) == len(indices)
    for port, index in zip(ports, indices, strict=True):
        expected = Bloq.wire_symbol(source, source_register, index).json_dict()
        assert port["symb_cls"] == expected["symb_cls"]
        assert port["symb_attributes"] == expected["symb_attributes"]


def test_parent_package_reflection_is_owned_by_documentation_metadata() -> None:
    from qualtran.bloqs.for_testing import TestAtom

    package, name = TestAtom._class_name_in_pkg_().rsplit(".", 1)
    metadata = ComponentDocumentation(component_type=TestAtom)
    assert metadata.component_type is TestAtom
    assert metadata.import_line == f"from {package} import {name}"
    assert metadata.examples == () and metadata.call_graph_example is None
