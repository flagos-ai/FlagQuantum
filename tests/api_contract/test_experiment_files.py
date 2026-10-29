"""Portable experiment workflows and hostile-file boundaries; CPU only."""

from __future__ import annotations

import hashlib
import inspect
import json
import struct
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import SerializationError
from flagquantum.runtime import _experiment_codec as codec

pytestmark = pytest.mark.unit
OPTIONS = fq.ExecutionOptions(shots=128, seed=42, device="cpu", backend="pytorch")


@pytest.fixture
def bell():
    program = fq.compile(fq.Circuit(2).h(0).cx(0, 1))
    return program, fq.run(program, outputs=fq.counts(), options=OPTIONS)


def contents(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def rewrite(path, members, *, checksums=False):
    if checksums:
        manifest = json.loads(members["manifest.json"])
        manifest["members"] = {
            name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()
            if name != "manifest.json"
        }
        members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_bell_save_load_is_data_only(tmp_path, bell):
    program, result = bell
    path = tmp_path / "bell.fqx"
    with (
        patch("flagquantum._api.run", side_effect=AssertionError("I/O must not run")),
        patch(
            "flagquantum._api.compile",
            side_effect=AssertionError("I/O must not compile"),
        ),
        patch("torch.multinomial", side_effect=AssertionError("I/O must not sample")),
    ):
        fq.save(path, program, result=result)
        restored = fq.load(path)
    assert restored.program.to_json() == program.to_json()
    assert restored.result.counts == result.counts
    assert restored.result.plan is None
    assert restored.execution_plan["identity"] == result.plan.identity
    assert sum(restored.result.counts[0].values()) == 128
    repeated = fq.run(restored.program, outputs=fq.counts(), options=OPTIONS)
    assert repeated.counts == result.counts
    assert not list(tmp_path.glob(".fqx-*.tmp"))


def test_program_only_round_trip_and_resave(tmp_path, bell):
    program, result = bell
    path = tmp_path / "program.FQX"
    fq.save(path, program)
    restored = fq.load(path)
    assert restored.result is None and restored.execution_plan is None
    assert restored.program.content_hash == program.content_hash
    fq.save(tmp_path / "run.fqx", result.plan, result=result)
    experiment = fq.load(tmp_path / "run.fqx")
    assert experiment.program.content_hash == result.plan.program_fingerprint
    fq.save(tmp_path / "copy.fqx", experiment)
    assert fq.load(tmp_path / "copy.fqx").result.counts == result.counts
    assert experiment.execution_plan == fq.load(tmp_path / "copy.fqx").execution_plan
    with pytest.raises(TypeError, match="omitted"):
        fq.save(tmp_path / "ambiguous.fqx", experiment, result=result)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_dtype_shape_values_and_owned_cpu_storage(tmp_path, bell, dtype):
    program, result = bell
    tensor = torch.tensor([[0, 1], [2, 3]], dtype=dtype).T
    recorded = replace(
        result,
        state=tensor,
        metrics={
            "complex": 2,
            "empty": torch.empty((0, 3), dtype=dtype),
            "scalar": torch.ones((), dtype=dtype),
        },
    )
    path = tmp_path / "typed.fqx"
    fq.save(path, program, result=recorded)
    loaded = fq.load(path).result
    assert loaded.state.dtype == dtype and loaded.state.shape == tensor.shape
    assert torch.equal(loaded.state, tensor)
    assert loaded.state.device.type == "cpu" and not loaded.state.requires_grad
    assert loaded.metrics["empty"].shape == (0, 3)
    assert loaded.metrics["scalar"].shape == ()
    tensor.fill_(0)
    assert loaded.state[1, 1].item() == torch.tensor(3, dtype=dtype).item()


def test_measurement_batches_qubit_order_and_diagnostics(tmp_path, bell):
    program, result = bell
    recorded = replace(
        result,
        measurements=(
            fq.MeasurementResult(
                kind="counts",
                qubits=(1, 0),
                value=[{0: 3, "11": 5}, {"01": 8}],
                shots=8,
                metadata={"seed": 7},
                statistics={"draws": [8, 8]},
            ),
        ),
        provenance={"distribution_semantics": "single_device_fast_path"},
        metrics={"loss": torch.tensor(0.5, requires_grad=True)},
    )
    fq.save(tmp_path / "batch.fqx", program, result=recorded)
    loaded = fq.load(tmp_path / "batch.fqx").result
    assert loaded.measurements[0].qubits == (1, 0)
    assert loaded.measurements[0].value == recorded.measurements[0].value
    assert loaded.measurements[0].statistics == {"draws": [8, 8]}
    assert loaded.provenance == recorded.provenance
    assert loaded.metrics["loss"].item() == 0.5
    assert not loaded.metrics["loss"].requires_grad


def test_mutated_program_or_other_plan_cannot_claim_a_result(tmp_path, bell):
    program, result = bell
    changed = fq.compile(fq.Circuit(2).h(0).cx(0, 1).x(1))
    with pytest.raises(SerializationError, match="differs"):
        fq.save(tmp_path / "wrong.fqx", changed, result=result)
    other = fq.plan(
        program, outputs=fq.counts(), options=fq.ExecutionOptions(shots=64, seed=11)
    )
    with pytest.raises(SerializationError, match="identities differ"):
        fq.save(tmp_path / "plan.fqx", other, result=result)
    with pytest.raises(SerializationError, match="retained"):
        fq.save(tmp_path / "no-plan.fqx", program, result=replace(result, plan=None))
    assert not list(tmp_path.iterdir())


def test_existing_destination_atomic_failure_and_explicit_replace(tmp_path, bell):
    program, result = bell
    path = tmp_path / "keep.fqx"
    path.write_bytes(b"existing user file")
    with pytest.raises(FileExistsError):
        fq.save(path, program)
    with pytest.raises(SerializationError, match="native handles"):
        fq.save(
            path,
            program,
            result=replace(result, metrics={"handle": object()}),
            overwrite=True,
        )
    assert path.read_bytes() == b"existing user file"
    assert not list(tmp_path.glob(".fqx-*.tmp"))
    fq.save(path, program, result=result, overwrite=True)
    assert fq.load(path).result.counts == result.counts
    with pytest.raises(SerializationError, match="extension"):
        fq.save(tmp_path / "wrong.json", program)
    with pytest.raises(TypeError, match="bool"):
        fq.save(tmp_path / "wrong.fqx", program, overwrite=1)


def test_content_identification_and_deterministic_container(tmp_path, bell):
    program, result = bell
    first, second = tmp_path / "a.fqx", tmp_path / "b.fqx"
    fq.save(first, program, result=result)
    fq.save(second, program, result=result)
    assert first.read_bytes() == second.read_bytes()
    renamed = first.rename(tmp_path / "renamed.zip")
    assert fq.load(renamed).result.counts == result.counts
    with pytest.raises(FileNotFoundError):
        fq.load(tmp_path / "missing.fqx")


@pytest.mark.parametrize(
    "attack",
    [
        "version",
        "checksum",
        "missing",
        "unknown",
        "path",
        "duplicate_json",
        "bad_tensor",
        "unreferenced",
        "duplicate_zip",
        "truncated",
        "compressed_data",
        "plan",
        "circuit_hash",
    ],
)
def test_corrupt_or_hostile_archive_fails_closed(tmp_path, bell, attack):
    program, result = bell
    path = tmp_path / "attack.fqx"
    fq.save(path, program, result=result)
    data = contents(path)
    if attack == "truncated":
        path.write_bytes(path.read_bytes()[:-10])
    elif attack == "compressed_data":
        with zipfile.ZipFile(path) as archive:
            member = archive.getinfo("circuit.json")
        raw = bytearray(path.read_bytes())
        name_size, extra_size = struct.unpack_from(
            "<HH", raw, member.header_offset + 26
        )
        offset = member.header_offset + 30 + name_size + extra_size
        raw[offset] = 0xFF  # Invalid DEFLATE block type, before JSON parsing.
        path.write_bytes(raw)
    elif attack == "duplicate_zip":
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("circuit.json", data["circuit.json"])
    elif attack == "version":
        manifest = json.loads(data["manifest.json"])
        manifest["version"] = "99.0"
        data["manifest.json"] = json.dumps(manifest).encode()
        rewrite(path, data)
    elif attack == "checksum":
        data["circuit.json"] += b" "
        rewrite(path, data)
    elif attack == "missing":
        del data["circuit.json"]
        rewrite(path, data)
    elif attack in ("unknown", "path", "unreferenced"):
        data[
            {
                "unknown": "unknown.json",
                "path": "../escape.py",
                "unreferenced": "arrays/9999.bin",
            }[attack]
        ] = b"no execution"
        rewrite(path, data, checksums=True)
    elif attack == "duplicate_json":
        data["manifest.json"] = b'{"schema":"x","schema":"y"}'
        rewrite(path, data)
    elif attack == "bad_tensor":
        payload = json.loads(data["result.json"])

        def change(value):
            if isinstance(value, dict):
                if value.get("kind") == "tensor":
                    value["shape"] = [2**30]
                    return True
                return any(change(item) for item in value.values())
            return isinstance(value, list) and any(change(item) for item in value)

        assert change(payload)
        data["result.json"] = json.dumps(payload).encode()
        rewrite(path, data, checksums=True)
    elif attack == "plan":
        payload = json.loads(data["result.json"])
        payload["execution_plan"]["identity"] = "0" * 64
        data["result.json"] = json.dumps(payload).encode()
        rewrite(path, data, checksums=True)
    else:
        payload = json.loads(data["manifest.json"])
        payload["program_content_hash"] = "0" * 64
        data["manifest.json"] = json.dumps(payload).encode()
        rewrite(path, data)
    with (
        patch(
            "builtins.exec",
            side_effect=AssertionError("archive must never execute code"),
        ),
        pytest.raises(SerializationError),
    ):
        fq.load(path)
    assert not (tmp_path.parent / "escape.py").exists()


def test_zip_directory_preflight_rejects_excess_entries_before_zip_parsing(tmp_path):
    path = tmp_path / "directory.fqx"
    path.write_bytes(
        b"PK\x03\x04"
        + struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 65535, 65535, 0, 4, 0)
    )
    with (
        patch(
            "zipfile.ZipFile", side_effect=AssertionError("must reject before parsing")
        ),
        pytest.raises(SerializationError, match="bounded"),
    ):
        fq.load(path)


def test_limits_and_recursive_metadata_are_explicit(tmp_path, bell, monkeypatch):
    program, result = bell
    monkeypatch.setattr(codec, "MAX_TENSOR_BYTES", 3)
    with pytest.raises(SerializationError, match="tensor exceeds"):
        fq.save(tmp_path / "too-big.fqx", program, result=result)
    monkeypatch.undo()
    recursive = {}
    recursive["self"] = recursive
    with pytest.raises(SerializationError, match="nesting"):
        fq.save(
            tmp_path / "recursive.fqx",
            program,
            result=replace(result, metrics=recursive),
        )
    assert not list(tmp_path.glob(".fqx-*.tmp"))


def test_historical_compiler_is_data_and_does_not_weaken_live_plan_validation(
    tmp_path, bell, monkeypatch
):
    program, result = bell
    path = tmp_path / "history.fqx"
    fq.save(path, program, result=result)
    original = result.plan.identity
    monkeypatch.setattr(
        "flagquantum.runtime.execution_plan_contract.__version__", "future-sdk"
    )
    loaded = fq.load(path)
    assert (
        loaded.execution_plan["identity"] == original
        and loaded.result.counts == result.counts
    )
    assert loaded.result.plan is None
    with pytest.raises(ValueError, match="compiler"):
        fq.ExecutionPlan.from_dict(loaded.execution_plan)
    fq.save(tmp_path / "history-copy.fqx", loaded)
    assert fq.load(tmp_path / "history-copy.fqx").execution_plan["identity"] == original


def test_exact_authorized_signatures():
    contract = json.loads(
        (
            Path(__file__).parents[2] / "contracts/experiment-file-v1-candidate.json"
        ).read_text()
    )
    for name, signature in contract["signatures"].items():
        assert str(inspect.signature(getattr(fq, name))) == signature


def test_root_import_keeps_archive_codec_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flagquantum; "
            "assert 'flagquantum.runtime.experiment' not in sys.modules; "
            "assert 'flagquantum.runtime._experiment_codec' not in sys.modules",
        ],
        cwd=Path(__file__).parents[2],
        check=True,
        timeout=20,
    )
