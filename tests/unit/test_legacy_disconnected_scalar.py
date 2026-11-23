"""Reject detectable scalar loss in the unchanged legacy composite wire format."""

from dataclasses import replace
from importlib import import_module

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.formats.legacy_scalar_factories import scalar_phase_factories
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_unchanged_source_composite_roundtrip_loses_scalar_and_native_rejects_it() -> (
    None
):
    from qualtran.bloqs.basic_gates import GlobalPhase, SU2RotationGate

    source = SU2RotationGate(0.2, 0.3, -0.4, 0.1).decompose_bloq()
    codec = import_module("qualtran.serialization.bloq")
    message = codec.bloqs_to_proto(source, max_depth=1)
    restored = codec.bloqs_from_proto(message)[0]
    assert any(
        isinstance(instance.bloq, GlobalPhase) for instance in source.bloq_instances
    )
    assert not any(
        isinstance(instance.bloq, GlobalPhase) for instance in restored.bloq_instances
    )
    assert not torch.allclose(
        torch.as_tensor(source.tensor_contract()),
        torch.as_tensor(restored.tensor_contract()),
    )
    library = deserialize_component_library(message.SerializeToString())
    with pytest.raises(SerializationError, match="zero-port execution child"):
        resolve_component_program(
            library, 0, {**rotation_factories(), **scalar_phase_factories()}
        )


def test_zero_resource_dependency_does_not_create_an_execution_child() -> None:
    from qualtran.bloqs.basic_gates import SU2RotationGate

    library = _library(SU2RotationGate(0.2, 0.3, -0.4, 0.1).decompose_bloq(), 1)
    root = library.entries[0]
    library = replace(
        library,
        entries=(
            replace(root, counts=tuple((child, 0) for child, _ in root.counts)),
            *library.entries[1:],
        ),
    )
    resolve_component_program(
        library, 0, {**rotation_factories(), **scalar_phase_factories()}
    )
