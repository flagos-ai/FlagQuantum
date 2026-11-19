"""Check every frozen package export has one existing native domain owner."""

import ast
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_complete_surface_code_export_disposition() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/surface_code/__init__.py"
    )
    raw = source.read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "9f43f1b6ac684c0649707f3559a0c0d87dc69849"
    )
    owners = {
        "AlgorithmSummary": (
            "flagquantum.compiler._component_trace.resource_model",
            "ModeledResources",
        ),
        "PhysicalCostsSummary": ("physical_cost", "PhysicalResourceEstimate"),
        "PhysicalParameters": ("physical_parameters", "PhysicalParameters"),
        "LogicalErrorModel": ("error_model", "LogicalErrorModel"),
        "QECScheme": ("error_model", "SurfaceCodeScheme"),
        "MagicStateFactory": ("magic_state_factory", "MagicStateFactory"),
        "CCZ2TFactory": ("ccz_to_t_factory", "CCZToTFactory"),
        "FifteenToOne": ("fifteen_to_one", "FifteenToOneFactory"),
        "CompactDataBlock": ("data_layout", "SingleHallwayDataLayout"),
        "DataBlock": ("data_layout", "SurfaceCodeDataLayout"),
        "FastDataBlock": ("data_layout", "SquareDataLayout"),
        "IntermediateDataBlock": ("data_layout", "TwoRowDataLayout"),
        "SimpleDataBlock": ("data_layout", "RoutingDataLayout"),
        "MultiFactory": ("parallel_factory", "ParallelFactory"),
        "BeverlandEtAlRotationCost": ("rotation_synthesis", "BEVERLAND_ROTATION_COST"),
        "ConstantWithOverheadRotationCost": (
            "rotation_synthesis",
            "PhaseGradientRotationCost",
        ),
        "RotationCostModel": ("rotation_synthesis", "RotationCostModel"),
        "RotationLogarithmicModel": ("rotation_synthesis", "LogarithmicRotationCost"),
        "SevenDigitsOfPrecisionConstantCost": (
            "rotation_synthesis",
            "SEVEN_BIT_PHASE_GRADIENT_COST",
        ),
        "get_ccz2t_costs": ("physical_cost", "PhysicalCostModel"),
        "get_ccz2t_costs_from_error_budget": (
            "physical_search",
            "estimate_gidney_fowler_from_budget",
        ),
        "get_ccz2t_costs_from_grid_search": (
            "physical_search",
            "search_gidney_fowler_resources",
        ),
        "iter_ccz2t_factories": ("physical_search", "iter_ccz_factories"),
        "iter_simple_data_blocks": ("physical_search", "iter_routing_data_layouts"),
        "PhysicalCostModel": ("physical_cost", "PhysicalCostModel"),
    }
    names = [
        alias.name
        for node in ast.parse(raw).body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert len(names) == len(owners) == 25
    assert set(names) == set(owners)
    for module, name in owners.values():
        path = (
            module
            if module.startswith("flagquantum.")
            else f"flagquantum.qec.resources.{module}"
        )
        assert getattr(import_module(path), name) is not None
