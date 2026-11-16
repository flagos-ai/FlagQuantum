# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Wire layout adapted from Qualtran v0.7.0 protos/bloq.proto.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/bloq.proto

"""Private descriptors for the legacy component library interchange."""

from typing import Any

from ._legacy_protobuf import message_factory
from .legacy_arguments import _array_message, _complex_message, _integer_message
from .legacy_control_specs import _control_spec_message
from .legacy_curve_points import _curve_message
from .legacy_register_types import _register_type_message
from .legacy_registers import _register_message, _registers_message
from .legacy_symbolic_terms import _term_message
from .resource_annotations import _annotation_factory


def _component_argument_message() -> Any:
    values = (
        ("int_val", "INT64"),
        ("float_val", "DOUBLE"),
        ("string_val", "STRING"),
        ("sympy_expr", _term_message()),
        ("ndarray", _array_message()),
        ("subbloq", "INT32"),
        ("cirq_json_gzip", "BYTES"),
        ("qdata_type", _register_type_message()),
        ("register", _register_message()),
        ("registers", _registers_message()),
        ("ctrl_spec", _control_spec_message()),
        ("complex_val", _complex_message()),
        ("ec_point", _curve_message()),
    )
    return message_factory(
        "ComponentArgument",
        (("name", "STRING"), *values),
        oneof="val",
        oneof_fields=tuple(name for name, _ in values),
    )


def _component_message() -> Any:
    return message_factory(
        "ComponentDeclaration",
        (
            ("name", "STRING"),
            ("args", _component_argument_message()),
            ("registers", _registers_message()),
            ("t_complexity", _annotation_factory()),
        ),
        repeated=("args",),
    )


def _port_message() -> Any:
    instance = message_factory(
        "ComponentOccurrence", (("instance_id", "INT32"), ("bloq_id", "INT32"))
    )
    return message_factory(
        "ComponentPort",
        (
            ("bloq_instance", instance),
            ("dangling_t", "STRING"),
            ("register", _register_message()),
            ("index", "INT32"),
        ),
        oneof="binst",
        oneof_fields=("bloq_instance", "dangling_t"),
        repeated=("index",),
    )


def _component_library_message() -> Any:
    port = _port_message()
    connection = message_factory(
        "ComponentConnection", (("left", port), ("right", port))
    )
    row = message_factory(
        "ComponentLibraryRow",
        (
            ("bloq_id", "INT32"),
            ("decomposition", connection),
            ("bloq_counts", _integer_message()),
            ("bloq", _component_message()),
        ),
        repeated=("decomposition",),
        integer_maps=("bloq_counts",),
    )
    return message_factory(
        "ComponentLibrary", (("name", "STRING"), ("table", row)), repeated=("table",)
    )
