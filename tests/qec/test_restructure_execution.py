"""Circuit/execution relocation against trusted pre-004 records and results."""

from __future__ import annotations

import dataclasses
import importlib
import io
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from _retirement import (
    assert_metadata,
    assert_retired_payload,
    resolve,
)

import flagquantum.qec as qec

pytestmark = pytest.mark.unit
_FIXTURES = Path(__file__).with_name("fixtures")


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


def _declarations() -> dict[str, object]:
    from flagquantum.qec.execution.signatures import (
        _forced_signature,
        _inject_data_flip,
    )

    result = {}
    for name, code in (
        ("repetition", qec.RepetitionCode()),
        ("surface", qec.RotatedSurfaceCode()),
        ("steane", qec.SteaneCode()),
    ):
        bases = ("x", "z") if name == "steane" else ("z",)
        for basis in bases:
            memory = qec.build_memory_circuit(code, rounds=2, readout_basis=basis)
            noise = qec.PhenomenologicalNoise(data_flip=0.125, measurement_flip=0.25)
            context = qec.decoder_context_from_memory_circuit(memory, noise=noise)
            component = context.full_component()
            injected = _inject_data_flip(
                memory, round_index=1, wire=code.data_qubits[0]
            )
            result[name + ":" + basis] = {
                "memory": _plain(memory),
                "context": _plain(component),
                "forced_source": injected,
                "forced_signature": _plain(_forced_signature(memory, injected)),
                "matrix_dem": qec.DetectorErrorModel.from_code(
                    code, num_rounds=2, noise=noise
                ).to_stim_text(),
                "circuit_dem": context.dem.to_stim_text(),
            }
    return result


def _samples() -> dict[str, object]:
    result = {}
    for name, code in (
        ("repetition", qec.RepetitionCode()),
        ("surface", qec.RotatedSurfaceCode()),
        ("steane", qec.SteaneCode()),
    ):
        bases = ("x", "z") if name == "steane" else ("z",)
        for basis in bases:
            memory = qec.build_memory_circuit(code, rounds=2, readout_basis=basis)
            for rate in (0.0, 0.125, 1.0):
                noise = qec.PhenomenologicalNoise(data_flip=rate, measurement_flip=rate)
                key = f"{name}:{basis}:{rate}"
                result[key] = {
                    "measurements": _plain(
                        qec.sample_memory_measurements(
                            memory, noise=noise, shots=16, seed=53
                        )
                    ),
                    "dem": _plain(
                        qec.sample_memory_circuit(
                            memory, noise=noise, shots=16, seed=53
                        )
                    ),
                }
    for mode in ("compiled_lookup", "offline_pauli_frame", "runtime_decoder"):
        result[mode] = _plain(
            qec.run_repetition_memory_experiment(
                rounds=3,
                shots=8,
                seed=53,
                strategy="trajectory",
                feedback_mode=mode,
                error_schedule=qec.ErrorSchedule((qec.ErrorEvent(1, 1),)),
            )
        )
    return result


def _resolve(path: str) -> object:
    return resolve(path)


def test_declarations_layouts_contexts_and_forced_signatures_match_pre_move() -> None:
    expected = json.loads((_FIXTURES / "restructure_004_execution.json").read_text())
    assert _declarations() == expected["declarations"]


def test_sampling_and_feedback_match_pre_move() -> None:
    stim = pytest.importorskip("stim")
    expected = json.loads((_FIXTURES / "restructure_004_execution.json").read_text())
    if (stim.__version__, platform.machine()) != (
        expected["stim_version"],
        expected["machine"],
    ):
        pytest.skip(
            "Exact seeded fixture requires its recorded Stim version and machine architecture"
        )
    assert _samples() == expected["samples"]


def test_original_execution_pickle_is_retired_and_canonical_checkpoints_preserve_state() -> (
    None
):
    assert_retired_payload("restructure_004_globals.pkl")
    memory = qec.build_memory_circuit(qec.RepetitionCode(), rounds=2)
    context = qec.decoder_context_from_memory_circuit(
        memory, noise=qec.PhenomenologicalNoise(data_flip=0.125, measurement_flip=0.25)
    )
    values = (memory, context, context.full_component())
    for value in values:
        restored = pickle.loads(pickle.dumps(value, protocol=4))
        assert type(restored) is type(value)
        assert _plain(restored) == _plain(value)
    expected = json.loads((_FIXTURES / "restructure_004_metadata.json").read_text())
    for path, metadata in expected.items():
        assert_metadata(path, metadata[0], metadata[1])
    checkpoint = io.BytesIO()
    torch.save(values, checkpoint)
    checkpoint.seek(0)
    allowed = set()

    def allow_records(value: object) -> None:
        if dataclasses.is_dataclass(value):
            allowed.add(type(value))
            for field in dataclasses.fields(value):
                allow_records(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                allow_records(item)

    allow_records(values)
    with torch.serialization.safe_globals(list(allowed)):
        restored = torch.load(checkpoint, weights_only=True)
    assert _plain(restored) == _plain(values)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("circuit", "circuits.memory"),
        ("context", "execution.context"),
        ("sampling", "execution.sampling"),
        ("repetition", "execution.repetition"),
    ),
)
def test_execution_imports_use_canonical_owners(old: str, new: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flagquantum.qec." + old)
    owner = importlib.import_module("flagquantum.qec." + new)
    for name in owner.__all__:
        assert getattr(owner, name) is getattr(qec, name)
    if old == "circuit":
        script = r"""
import importlib.abc
import sys
import flagquantum.qec
blocked = ("flagquantum.compiler", "flagquantum.runtime", "flagquantum.qec.execution", "flagquantum.qec.dem")
for name in tuple(sys.modules):
    if name.startswith(blocked) or name.startswith("flagquantum.qec.circuits"):
        del sys.modules[name]
class NoExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(blocked):
            raise AssertionError("Declaration imported execution: " + fullname)
sys.meta_path.insert(0, NoExecution())
from flagquantum.qec.circuits.memory import build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
memory = build_memory_circuit(RepetitionCode(), rounds=3)
assert len(memory.detectors) == 8
assert len(memory.measurement_refs) == 9
assert not any(name.startswith(blocked) for name in sys.modules)
print("Declaration-only construction passed")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.stdout.strip() == "Declaration-only construction passed"
