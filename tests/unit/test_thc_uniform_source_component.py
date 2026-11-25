"""Retain the full original THC uniform graph and its physical refusal."""

from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_thc_preparation_factories import (
    thc_preparation_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.primitives.identity import register_identity_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def pinned_thc_source() -> None:
    from qualtran.bloqs.chemistry.thc import prepare

    for name, expected in (
        ("prepare.py", "f93891e72790d0e86f1b73c6a1c1c49089d2550c"),
        ("prepare_test.py", "8f3da8352c3cd76aff25291b1f9900380a3a9660"),
    ):
        raw = (Path(prepare.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


@pytest.mark.parametrize("m,n", ((2, 4), (4, 6), (8, 8), (10, 4)))
@pytest.mark.parametrize("inverse", (False, True))
def test_literal_complete_source_calls_operands_and_physical_refusal(
    m: int, n: int, inverse: bool
) -> None:
    from qualtran.bloqs.arithmetic.comparison import GreaterThanConstant
    from qualtran.bloqs.chemistry.thc.prepare import UniformSuperpositionTHC

    original = UniformSuperpositionTHC(m, n)
    source = original.adjoint() if inverse else original
    assert len(source.decompose_bloq().bloq_instances) == 35
    for depth in (0, 1):
        record = resolve_component_program(
            _library(source, depth), 0, thc_preparation_factories()
        )
        assert not verify_program_record(record)
        opaque = {f.name for f in record.functions if not f.blocks}
        assert opaque
        terminals = atomic_leaves(record) | opaque
        for width in {1, m.bit_length()}:
            identity = register_identity_program(width)
            terminals.update(
                f.name
                for f in record.functions
                if f.registers == identity.functions[0].registers
                and matches_component_definition(record, f.name, identity)
            )
        pairs = compare_layers(record, source, terminals)
        assert any(
            isinstance(
                getattr(pairs[name], "subbloq", pairs[name]), GreaterThanConstant
            )
            for name in opaque
            if name in pairs
        )
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_used_workspace(record)
