"""Frozen modular package aliases enter the authoritative constructor paths."""

from collections import Counter
from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import sympy
import torch

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_addition_factories import (
    modular_addition_factories,
)
from flagquantum.compiler.formats.legacy_modular_subtraction_factories import (
    modular_subtraction_factories,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_addition_component import bind_case

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_PREFIX = "qualtran.bloqs.mod_arithmetic."


def _factories() -> dict[str, ComponentFactory]:
    return {**modular_addition_factories(), **modular_subtraction_factories()}


def _source(kind: str, cv: int = 1, symbolic: bool = False) -> Any:
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic import CModAddK, CModSub, ModAdd, ModSub

    width = sympy.Symbol("n") if symbolic else 2
    if kind == "CModAddK":
        return CModAddK(1, 2, width)
    if kind == "ModAdd":
        return ModAdd(width, 2)
    dtype = QUInt(width)
    return CModSub(dtype, 2, cv) if kind == "CModSub" else ModSub(dtype, 2)


def _resolve(source: Any, alias: bool) -> ProgramRecord:
    from qualtran.protos.bloq_pb2 import BloqLibrary
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    message = bloqs_to_proto(source, max_depth=0)
    if alias:
        # Construct an explicit legacy payload; neither reader nor source is patched.
        message.table[0].bloq.name = _PREFIX + type(source).__name__
    payload = message.SerializeToString()
    decoded = BloqLibrary()
    decoded.ParseFromString(payload)
    if not alias and type(source).__name__ in {"ModSub", "CModSub"}:
        with pytest.raises(ValueError, match="Unable to find a Bloq"):
            bloqs_from_proto(decoded)
    else:
        assert bloqs_from_proto(decoded)[0] == source
    library = deserialize_component_library(payload)
    assert (
        deserialize_component_library(serialize_component_library(library)) == library
    )
    program = resolve_component_program(library, 0, _factories())
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize("kind", ("CModAddK", "ModAdd", "ModSub", "CModSub"))
@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("cv", (0, 1))
def test_same_source_bytes_alias_and_canonical_constructor_paths(
    kind: str, symbolic: bool, cv: int
) -> None:
    source = _source(kind, cv, symbolic)
    factories = _factories()
    module = "mod_addition" if kind in {"CModAddK", "ModAdd"} else "mod_subtraction"
    assert factories[_PREFIX + kind] is factories[_PREFIX + module + "." + kind]
    if kind == "ModAdd":
        # Frozen wire omits Montgomery modulus metadata for both spellings.
        for alias in (False, True):
            with pytest.raises(SerializationError, match="signature"):
                _resolve(source, alias)
        return
    canonical, legacy = _resolve(source, False), _resolve(source, True)
    assert canonical == legacy
    if symbolic:
        assert not legacy.functions[0].blocks
        return
    bound, ports, _ = bind_case(legacy)
    expected, _, _ = bind_case(canonical)
    assert bound == expected
    from tests.unit.test_modular_addition_component import source_stream as add_stream
    from tests.unit.test_modular_subtraction_component import (
        source_stream as sub_stream,
    )
    from tests.unit.test_register_subtraction import assert_source

    stream = add_stream if kind == "CModAddK" else sub_stream
    assert_source(bound.circuit.instructions, stream(source, bound, ports))
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == counts["reset"] == model.measurement


@pytest.mark.parametrize(
    "kind,cv", (("CModAddK", 1), ("ModSub", 1), ("CModSub", 0), ("CModSub", 1))
)
def test_alias_full_complex_branches_and_zero_private_work(kind: str, cv: int) -> None:
    source = _source(kind, cv)
    bound, _, logical = bind_case(_resolve(source, True))
    gates = bound.circuit.instructions
    n = max((w for gate in gates for w in gate.wires), default=logical - 1) + 1
    scratch = n - logical
    before, after = [], []
    controls = (0, 1) if kind.startswith("C") else (1,)
    for ctrl, x, y in product(
        controls, range(2), range(2) if kind != "CModAddK" else (0,)
    ):
        if kind == "CModAddK":
            old = (ctrl << 2) | x
            new = (ctrl << 2) | ((x + 1) % 2 if ctrl else x)
        else:
            old = (x << 2) | y
            new = (x << 2) | ((y - x) % 2 if ctrl == cv else y)
            if kind == "CModSub":
                old |= ctrl << 4
                new |= ctrl << 4
        before.append(old << scratch)
        after.append(new << scratch)
    raw = torch.arange(1, len(before) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 2 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    target = torch.zeros_like(initial)
    initial[before], target[after] = amplitudes, amplitudes
    measurements = sum(g.name == "measure" for g in gates)
    basis = tuple(
        tuple((label >> (n - 1 - i)) & 1 for i in range(n)) for label in before
    )
    mapped, events = certified_and_channel_map(gates, basis)
    expected = tuple(
        tuple((label >> (n - 1 - i)) & 1 for i in range(n)) for label in after
    )
    assert mapped == expected and events == measurements
    histories = (
        tuple(product((0, 1), repeat=measurements))
        if measurements <= 6
        else (
            (0,) * measurements,
            (1,) * measurements,
            tuple(i % 2 for i in range(measurements)),
        )
    )
    for history in histories:
        actual = forced_branch(initial.clone(), gates, history)
        torch.testing.assert_close(
            actual, target / (2 ** (measurements / 2)), atol=1e-13, rtol=0
        )


def test_complete_package_exports_and_locked_native_environment() -> None:
    from importlib import import_module
    from importlib.metadata import version

    import kickmix
    from qualtran.bloqs import mod_arithmetic

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    exports = {
        "mod_addition": ("CModAdd", "CModAddK", "CtrlScaleModAdd", "ModAdd", "ModAddK"),
        "mod_division": ("KaliskiModInverse",),
        "mod_multiplication": ("CModMulK", "DirtyOutOfPlaceMontgomeryModMul", "ModDbl"),
        "mod_subtraction": ("CModNeg", "CModSub", "ModNeg", "ModSub"),
    }
    for module, names in exports.items():
        owner = import_module(_PREFIX + module)
        for name in names:
            assert getattr(mod_arithmetic, name) is getattr(owner, name)
    assert sum(map(len, exports.values())) == 13
    for path, digest in (
        (Path(mod_arithmetic.__file__), "774aa44d6c868ebc8eaa08386ac9b318b0d898de"),
        (
            Path(mod_arithmetic.__file__).parents[2] / "serialization/resolver_dict.py",
            "b10adb30b493afda81fcff2e330618415df655a2",
        ),
    ):
        raw = path.read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
