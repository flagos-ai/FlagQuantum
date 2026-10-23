"""DEM package relocation against trusted pre-005 identifiers and exact results."""

from __future__ import annotations

import dataclasses
import importlib
import json
import pickle
import runpy
from pathlib import Path

import pytest
import torch

import flagquantum.qec as qec

pytestmark = pytest.mark.unit
_FIXTURES = Path(__file__).with_name("fixtures")
_OWNERS = {
    "dem": "model",
    "dem_text": "text",
    "dem_alternatives": "alternatives",
    "dem_matrices": "matrices",
    "dem_construction": "construction",
    "dem_circuit": "circuit",
    "chunks": "chunks",
    "decoding_graph": "graph",
}


def _plain(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "values": value.tolist()}
    if dataclasses.is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _scenarios() -> dict[str, object]:
    model = qec.DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L0\nerror(0.25) D0 D1\ndetector D0\ndetector D1\ndetector D2\nlogical_observable L0\n"
    )
    exclusive = qec.DetectorErrorModel(
        num_detectors=2,
        num_observables=1,
        errors=(
            qec.DemError(0.125, (0,), (0,), error_id=7),
            qec.DemError(0.25, (1,), error_id=7),
            qec.DemError(0.0625, (0, 1)),
        ),
    )
    duplicate = qec.DetectorErrorModel(
        num_detectors=1,
        num_observables=0,
        errors=(qec.DemError(0.125, (0,)), qec.DemError(0.25, (0,))),
    )
    result = {}
    for name, item in (("independent", model), ("exclusive", exclusive)):
        result[name] = {
            "model": _plain(item),
            "detectors": _plain(item.detector_error_matrix()),
            "observables": _plain(item.observables_flips_matrix()),
            "detector_rates": _plain(item.detector_rates()),
            "observable_rates": _plain(item.observable_rates()),
            "sample": _plain(item.dem_sampling(shots=16, seed=57)),
            "ids": _plain(item.error_ids),
            "stated_ids": _plain(item.stated_error_ids()),
            "groups": _plain(item.exclusive_groups()),
        }
    result["text"] = model.to_stim_text()
    result["graph"] = _plain(qec.DecodingGraph.from_detector_error_model(model))
    circuit = qec.circuit_from_detector_error_model(model)
    result["realized"] = qec.detector_error_model_from_circuit(
        circuit, num_detectors=model.num_detectors
    ).to_stim_text()
    for rule in qec.DemMergeRule:
        result["merge:" + rule.value] = _plain(
            duplicate.merge_duplicate_mechanisms(rule=rule)
        )
    for name, code in (
        ("repetition", qec.RepetitionCode()),
        ("surface", qec.RotatedSurfaceCode()),
    ):
        noise = qec.PhenomenologicalNoise(data_flip=0.125, measurement_flip=0.25)
        memory = qec.build_memory_circuit(code, rounds=2)
        result[name + ":matrices"] = qec.DetectorErrorModel.from_code(
            code, noise=noise, num_rounds=2
        ).to_stim_text()
        result[name + ":memory"] = qec.DetectorErrorModel.from_memory_circuit(
            memory, noise=noise
        ).to_stim_text()
    helpers = runpy.run_path(str(Path(__file__).with_name("test_dem_chunks.py")))
    layout = helpers["_layout"]()
    specs = qec.DemChunksSpec(layout=layout, window=2)
    chunks = qec.dem_chunks_from_spec(specs)
    result["chunks"] = _plain(chunks)
    result["stitched"] = _plain(qec.dem_stitch_all(chunks))
    result["closed"] = _plain(qec.dem_close_all(chunks))
    for text in ("error(2) D0", "error(0.1) BAD", "repeat -1 { error(0.1) D0 }"):
        try:
            qec.DetectorErrorModel.from_stim_text(text)
        except (ValueError, TypeError) as error:
            result["refusal:" + text] = [type(error).__name__, str(error)]
    return result


def _resolve(path: str) -> object:
    parts = path.split(".")
    value = importlib.import_module(".".join(parts[:3]))
    for part in parts[3:]:
        value = getattr(value, part)
    return value


def test_trusted_old_dem_pickles_keep_exact_objects() -> None:
    # Repository-owned fixture captured before relocation; never external input.
    restored = pickle.loads((_FIXTURES / "restructure_005_globals.pkl").read_bytes())
    for path, value in restored["globals"].items():
        assert value is _resolve(path), path
        assert pickle.loads(pickle.dumps(value, protocol=4)) is value, path
    for path, value in restored["instances"].items():
        assert type(value) is _resolve(path)
        assert type(pickle.loads(pickle.dumps(value))) is type(value)


def test_dem_metadata_and_historical_namespace_remain_identical() -> None:
    expected = json.loads((_FIXTURES / "restructure_005_metadata.json").read_text())
    for path, metadata in expected["definitions"].items():
        value = _resolve(path)
        if isinstance(value, property):
            value = value.fget
        assert [value.__module__, value.__qualname__] == metadata, path
    for old, names in expected["namespaces"].items():
        historical = importlib.import_module("flagquantum.qec." + old)
        for name in names:
            assert hasattr(historical, name), (old, name)


def test_dem_numerical_serialization_and_seam_baseline() -> None:
    expected = json.loads((_FIXTURES / "restructure_005_results.json").read_text())
    assert _scenarios() == expected


def test_both_rotated_identities_and_registries_survive_dem_package_imports() -> None:
    from flagquantum.qec import codes, surface

    assert qec.RotatedSurfaceCode is codes.RotatedSurfaceCode
    assert qec.RotatedSurfaceCode is not surface.RotatedSurfaceCode
    assert type(qec.get_code("rotated_surface", distance=3)) is qec.RotatedSurfaceCode
    assert qec.decoder_names() == (
        "belief_propagation",
        "belief_propagation_osd",
        "minimum_weight_matching",
        "pymatching",
        "sliding_window_matching",
    )


def test_historical_dem_imports_resolve_the_canonical_definitions() -> None:
    expected = json.loads((_FIXTURES / "restructure_005_metadata.json").read_text())
    for path in expected["definitions"]:
        parts = path.split(".")
        if len(parts) != 4:
            continue
        owner = importlib.import_module("flagquantum.qec.dem." + _OWNERS[parts[2]])
        assert _resolve(path) is getattr(owner, parts[3]), path


def test_model_factories_do_not_import_execution_after_legacy_bootstrap() -> None:
    # The root keeps its historical eager facade. Verify the new model boundary
    # after that bootstrap, in a fresh process without changing caller state.
    import subprocess
    import sys

    script = r"""
import importlib
import importlib.abc
import sys
import flagquantum.qec

blocked = (
    "flagquantum.compiler", "flagquantum.runtime",
    "flagquantum.qec.execution", "flagquantum.qec.dem.construction",
    "flagquantum.qec.dem_construction",
)
reload = (
    "flagquantum.qec.dem.model", "flagquantum.qec.dem.text",
    "flagquantum.qec.dem.alternatives", "flagquantum.qec.dem.matrices",
)
for name in tuple(sys.modules):
    if name in reload or any(name == p or name.startswith(p + ".") for p in blocked):
        del sys.modules[name]

class RefuseExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
            raise AssertionError("model crossed the execution boundary: " + fullname)
        return None

sys.meta_path.insert(0, RefuseExecution())
owner = importlib.import_module("flagquantum.qec.dem.model")
from flagquantum.qec.codes import SteaneCode
from flagquantum.qec.noise import PhenomenologicalNoise
text = "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
model = owner.DetectorErrorModel.from_stim_text(text)
assert model.detector_rates().tolist() == [0.125]
capacity = owner.DetectorErrorModel.from_code(
    SteaneCode(), noise=PhenomenologicalNoise(data_flip=0.125), num_rounds=2,
)
assert capacity.num_detectors == 12
assert not any(n == p or n.startswith(p + ".") for n in sys.modules for p in blocked)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
