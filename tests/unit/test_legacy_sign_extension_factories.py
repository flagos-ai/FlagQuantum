"""Signed conversion constructors preserve directional ownership and source fanout."""

from collections import Counter
from importlib import import_module
from typing import Any

import pytest
import sympy
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fanout_factories import fanout_factories
from flagquantum.compiler.formats.legacy_sign_extension_factories import (
    sign_extension_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import _factories, _library
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _resolve(source: Any, depth: int = 1) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, depth),
        0,
        {**_factories(), **sign_extension_factories(), **fanout_factories()},
    )
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize("encoding", ("QInt", "QIntOnesComp"))
@pytest.mark.parametrize("width,extra", ((2, 1), (2, 3), (3, 2), (8, 8)))
def test_sign_constructors_named_and_direct_source_gates_roundtrip(
    encoding: str, width: int, extra: int
) -> None:
    source_type = getattr(import_module("qualtran"), encoding)
    source = import_module(
        "qualtran.bloqs.arithmetic.conversions.sign_extension"
    ).SignExtend(source_type(width), source_type(width + extra))
    ports = (tuple(range(width)),)
    work = tuple(range(width + 1, width + extra + 1))
    for direct in (False, True):
        original = source.decompose_bloq() if direct else source
        forward = bind_register_program(
            _resolve(original, 2),
            ports,
            clean=work,
            allow_workspace_outputs=True,
        )
        inverse = source.adjoint()
        back = bind_register_program(
            _resolve(inverse.decompose_bloq() if direct else inverse, 2),
            forward.outputs,
            input_ownership=tuple((w, False) for w in work),
        )
        assert forward.outputs == ((*work, *ports[0]),) and back.outputs == ports
        for bound, upstream, before, after in (
            (forward, source, ports, forward.outputs),
            (back, inverse, forward.outputs, ports),
        ):
            assert_source(
                bound.circuit.instructions,
                reference_arithmetic_gates(
                    upstream,
                    {"x": before[0]},
                    work if upstream is source else (),
                    {},
                    expected_outputs={"y": after[0]},
                ),
            )
            actual = Counter(g.name for g in bound.circuit.instructions)
            assert actual == {"cx": 2 * extra - 1}
            from qualtran.resource_counting import QECGatesCost, get_cost_value

            model = get_cost_value(upstream, QECGatesCost())
            assert (
                model.clifford == 1 and model.total_t_count() == model.measurement == 0
            )
        if width > 3:
            continue
        coeff = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
        amplitudes = torch.complex(coeff, 3 - 7 * coeff)
        amplitudes /= amplitudes.norm()
        initial = torch.zeros(1 << (width + extra + 1), dtype=torch.complex128)
        initial[:: 1 << extra] = amplitudes
        prepared = forced_branch(initial, forward.circuit.instructions, ())
        expected = torch.zeros_like(initial)
        raw = torch.arange(1 << (width + 1))
        expected[(raw << extra) | ((raw >> width) * ((1 << extra) - 1))] = amplitudes
        torch.testing.assert_close(prepared, expected, rtol=0, atol=3e-12)
        torch.testing.assert_close(
            forced_branch(prepared, back.circuit.instructions, ()),
            initial,
            rtol=0,
            atol=3e-12,
        )


def test_symbolic_sign_contracts_and_invalid_metadata() -> None:
    factories = sign_extension_factories()
    n = sympy.Symbol("n", positive=True, integer=True)
    for kind in ("SignExtend", "SignTruncate"):
        factory = factories[
            "qualtran.bloqs.arithmetic.conversions.sign_extension." + kind
        ]
        before, after = (RegisterType("signed", n), RegisterType("signed", n + 1))
        if kind == "SignTruncate":
            before, after = after, before
        result = factory("custom", None, {"inp_dtype": before, "out_dtype": after})
        assert isinstance(result, FunctionRecord) and not result.blocks
        assert (
            factory(
                "custom", result.registers, {"inp_dtype": before, "out_dtype": after}
            )
            == result
        )
        concrete = {
            "inp_dtype": RegisterType("signed", 3 if kind == "SignTruncate" else 2),
            "out_dtype": RegisterType("signed", 2 if kind == "SignTruncate" else 3),
        }
        assert isinstance(factory("custom", None, concrete), ProgramRecord)
        invalid: tuple[dict[str, ResolvedArgument], ...] = (
            {},
            {"inp_dtype": "bad", "out_dtype": after},
            {"inp_dtype": before, "out_dtype": "bad"},
            {
                "inp_dtype": RegisterType("uint", 2),
                "out_dtype": RegisterType("uint", 3),
            },
            {
                "inp_dtype": RegisterType("signed", 2),
                "out_dtype": RegisterType("ones_complement", 3),
            },
        )
        for args in invalid:
            with pytest.raises(SerializationError):
                factory("bad", None, args)
        with pytest.raises(SerializationError, match="signature"):
            factory("bad", (), concrete)


def test_source_sympy_numeric_difference_remains_symbolic() -> None:
    from qualtran import QInt
    from qualtran.bloqs.arithmetic.conversions.sign_extension import (
        SignExtend,
        SignTruncate,
    )

    n = sympy.Symbol("n", positive=True, integer=True)
    for source in (
        SignExtend(QInt(n + 1), QInt(n)),
        SignTruncate(QInt(n), QInt(n + 1)),
    ):
        result = _resolve(source, 0)
        assert result.functions[0].blocks == ()
