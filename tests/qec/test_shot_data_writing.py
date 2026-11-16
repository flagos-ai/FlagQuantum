"""User-provided records are admitted once and published without replacement."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal, NoReturn

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec.execution.shot_data import write_shot_data_file
from flagquantum.simulation.stabilizer import _shot_data_write as owner
from flagquantum.simulation.stabilizer import engine
from flagquantum.simulation.stabilizer.engine import StabilizerDependencyError

pytestmark = pytest.mark.unit


class FakeWriter(ModuleType):
    """Independent b8 consumer of the exact bytes handed to the codec boundary."""

    def __init__(self) -> None:
        super().__init__("fake_writer")
        self.calls: list[tuple[bytes, int, bool, tuple[int, int, int]]] = []
        self.paths: list[Path] = []
        self.fail = False
        self.corrupt = False
        self.race_target: Path | None = None
        self.mutate: torch.Tensor | None = None

    def write_shot_records(
        self,
        data: bytearray,
        shots: int,
        packed: bool,
        path: str,
        format: str,
        nm: int,
        nd: int,
        no: int,
    ) -> None:
        assert format == "b8"
        self.calls.append((bytes(data), shots, packed, (nm, nd, no)))
        self.paths.append(Path(path))
        if self.mutate is not None:
            self.mutate.zero_()
        if self.fail:
            Path(path).write_bytes(b"partial")
            raise RuntimeError("controlled codec failure")
        if self.race_target is not None:
            self.race_target.write_bytes(b"winner")
        columns = nm + nd + no
        stride = (columns + 7) // 8
        output = bytearray(shots * stride)
        for shot in range(shots):
            for bit in range(columns):
                value = (
                    (data[shot * stride + bit // 8] >> (bit % 8)) & 1
                    if packed
                    else data[shot * columns + bit]
                )
                output[shot * stride + bit // 8] |= value << (bit % 8)
        Path(path).write_bytes(output if not self.corrupt else b"bad")


@pytest.fixture
def captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, FakeWriter]:
    backend = FakeWriter()
    monkeypatch.setattr(owner, "_backend", lambda: backend)
    return tmp_path / "events.b8", backend


def test_contract_preserves_approved_signature() -> None:
    import inspect

    from flagquantum.qec.execution import shot_data

    contract = json.loads(
        (
            Path(__file__).parents[2]
            / "contracts/qec-shot-data-writing-v1-candidate.json"
        ).read_text()
    )
    assert (
        contract["approval"]["proposal_sha256"]
        == "b22e6af76b65abd4a93ff97e9825197f83fe815b509b226f33b65b2818eeba83"
    )
    signature = inspect.signature(write_shot_data_file)
    assert tuple(signature.parameters) == (
        "data",
        "path",
        "format",
        "num_measurements",
        "num_detectors",
        "num_observables",
    )
    assert all(
        p.kind == inspect.Parameter.KEYWORD_ONLY for p in signature.parameters.values()
    )
    assert all(
        signature.parameters[n].default is None
        for n in ("num_measurements", "num_detectors", "num_observables")
    )
    assert shot_data.__all__ == (
        "read_shot_data_file",
        "iter_shot_data_file",
        "write_shot_data_file",
    )


@pytest.mark.parametrize("packed", [False, True])
def test_same_consumer_accepts_owned_bool_or_packed(
    captured: tuple[Path, FakeWriter], packed: bool
) -> None:
    path, backend = captured
    data = (
        torch.tensor([[0], [3]], dtype=torch.uint8)
        if packed
        else torch.tensor([[False, False], [True, True]])
    )
    backend.mutate = data
    write_shot_data_file(
        data=data, path=path, format="b8", num_detectors=1, num_observables=1
    )
    assert path.read_bytes() == b"\x00\x03"
    assert backend.calls == [
        (b"\x00\x03" if packed else b"\x00\x00\x01\x01", 2, packed, (0, 1, 1))
    ]
    assert all(not p.exists() for p in backend.paths)
    assert not bool(data.any())


def test_noncontiguous_input_is_copied(captured: tuple[Path, FakeWriter]) -> None:
    path, backend = captured
    data = torch.tensor([[False, True], [True, False], [False, True]]).T
    assert not data.is_contiguous()
    write_shot_data_file(data=data, path=path, format="b8", num_measurements=3)
    assert path.read_bytes() == b"\x02\x05"
    assert len(backend.calls) == 1


def test_empty_shot_table(captured: tuple[Path, FakeWriter]) -> None:
    path, backend = captured
    write_shot_data_file(
        data=torch.empty((0, 2), dtype=torch.bool),
        path=path,
        format="b8",
        num_measurements=2,
    )
    assert path.read_bytes() == b"" and backend.calls[0][1] == 0


@pytest.mark.parametrize(
    "dimensions,error",
    [
        ((None, None, None), ValueError),
        ((0, 0, 0), CapabilityError),
        ((True, 0, 0), TypeError),
        ((-1, 0, 0), ValueError),
        ((1, 1, 0), ValueError),
        ((1, 0, 1), ValueError),
        ((sys.maxsize + 1, 0, 0), CapabilityError),
    ],
)
def test_dimensions_refuse_before_backend(
    dimensions: tuple[int | None, int | None, int | None],
    error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> NoReturn:
        raise AssertionError("backend must not prepare")

    monkeypatch.setattr(owner, "_backend", forbidden)
    with pytest.raises(error):
        write_shot_data_file(
            data=torch.zeros((1, 1), dtype=torch.bool),
            path="missing/out",
            format="b8",
            num_measurements=dimensions[0],
            num_detectors=dimensions[1],
            num_observables=dimensions[2],
        )


@pytest.mark.parametrize(
    "data,error",
    [
        (torch.zeros((1, 2)), TypeError),
        (torch.zeros((2,), dtype=torch.bool), ValueError),
        (torch.zeros((1, 3), dtype=torch.bool), ValueError),
        (torch.tensor([[128]], dtype=torch.uint8), ValueError),
        (torch.empty((1, 2), dtype=torch.bool, device="meta"), CapabilityError),
        (torch.zeros((1, 2), dtype=torch.bool).to_sparse(), CapabilityError),
    ],
)
def test_tensor_refusals_before_backend(
    data: torch.Tensor, error: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> NoReturn:
        raise AssertionError("backend must not prepare")

    monkeypatch.setattr(owner, "_backend", forbidden)
    with pytest.raises(error):
        write_shot_data_file(
            data=data, path="missing/out", format="b8", num_detectors=2
        )


def test_ptb64_refuses_incomplete_group(captured: tuple[Path, FakeWriter]) -> None:
    path, backend = captured
    with pytest.raises(ValueError, match="multiple of 64"):
        write_shot_data_file(
            data=torch.zeros((63, 1), dtype=torch.uint8),
            path=path,
            format="ptb64",
            num_detectors=1,
        )
    assert not backend.calls and not path.exists()


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling"])
def test_existing_destination_is_never_replaced(
    captured: tuple[Path, FakeWriter], kind: str
) -> None:
    path, backend = captured
    original = path.parent / "original"
    original.write_bytes(b"keep")
    if kind == "file":
        path.write_bytes(b"keep")
    else:
        path.symlink_to(original if kind == "symlink" else path.parent / "absent")
    with pytest.raises(FileExistsError):
        write_shot_data_file(
            data=torch.zeros((1, 1), dtype=torch.uint8),
            path=path,
            format="b8",
            num_detectors=1,
        )
    assert original.read_bytes() == b"keep" and not backend.calls


@pytest.mark.parametrize("corrupt", [False, True])
def test_failed_or_incomplete_writer_cleans_staging(
    captured: tuple[Path, FakeWriter], corrupt: bool
) -> None:
    path, backend = captured
    backend.corrupt, backend.fail = corrupt, not corrupt
    with pytest.raises(OSError):
        write_shot_data_file(
            data=torch.zeros((1, 1), dtype=torch.uint8),
            path=path,
            format="b8",
            num_detectors=1,
        )
    assert not path.exists() and all(not p.exists() for p in backend.paths)
    assert list(path.parent.iterdir()) == []


def test_publication_race_keeps_winner(captured: tuple[Path, FakeWriter]) -> None:
    path, backend = captured
    backend.race_target = path
    with pytest.raises(FileExistsError):
        write_shot_data_file(
            data=torch.zeros((1, 1), dtype=torch.uint8),
            path=path,
            format="b8",
            num_detectors=1,
        )
    assert path.read_bytes() == b"winner" and all(not p.exists() for p in backend.paths)


def test_payload_limit_exact_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    needed = owner._principal_bytes(64, 9, True)
    monkeypatch.setattr(owner, "_MAX_PAYLOAD", needed)
    owner._check_budget(64, 9, True)
    monkeypatch.setattr(owner, "_MAX_PAYLOAD", needed - 1)
    with pytest.raises(CapabilityError, match="64 MiB"):
        owner._check_budget(64, 9, True)


def test_missing_sdk_retains_original_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> NoReturn:
        raise ModuleNotFoundError("No module named 'stim'", name="stim")

    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(engine, "import_module", missing)
    with pytest.raises(StabilizerDependencyError) as error:
        owner._backend()
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_broken_embedded_cannot_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> NoReturn:
        raise ImportError("controlled broken embedded module")

    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(owner, "import_module", missing)
    with pytest.raises(CapabilityError, match="embedded shot writer"):
        owner._backend()


def test_wrong_sdk_version_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    class WrongSDK(ModuleType):
        __version__: str = "1.15.0"

    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(engine, "_stim", lambda: WrongSDK("stim"))
    with pytest.raises(CapabilityError, match="1.16.0"):
        owner._backend()


@pytest.mark.parametrize("format", ["unknown", "", None, 12])
def test_invalid_format_prepares_no_backend(
    format: Literal["b8"], captured: tuple[Path, FakeWriter]
) -> None:
    path, backend = captured
    with pytest.raises(ValueError, match="shot format"):
        write_shot_data_file(
            data=torch.ones((1, 1), dtype=torch.bool),
            path=path,
            format=format,
            num_detectors=1,
        )
    assert not backend.calls


def test_output_budget_checked_before_copy_or_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(owner, "_MAX_OUTPUT", 0)

    def forbidden(*args: object) -> NoReturn:
        raise AssertionError("must refuse before copying or preparing a backend")

    monkeypatch.setattr(owner, "_owned_data", forbidden)
    monkeypatch.setattr(owner, "_backend", forbidden)
    with pytest.raises(CapabilityError, match="1 GiB"):
        write_shot_data_file(
            data=torch.ones((1, 1), dtype=torch.bool),
            path=tmp_path / "new",
            format="b8",
            num_measurements=1,
        )


def test_output_budget_exact_boundary(
    captured: tuple[Path, FakeWriter], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = captured
    monkeypatch.setattr(owner, "_MAX_OUTPUT", 1)
    write_shot_data_file(
        data=torch.tensor([[True]]), path=path, format="b8", num_measurements=1
    )
    assert path.read_bytes() == b"\x01"


def test_no_disk_space_refuses_before_codec(
    captured: tuple[Path, FakeWriter], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, backend = captured
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="free space"):
        write_shot_data_file(
            data=torch.tensor([[True]]), path=path, format="b8", num_measurements=1
        )
    assert not path.exists() and not backend.calls


@pytest.mark.parametrize(
    "path,error", [(b"bytes", TypeError), ("https://example.test/shots", ValueError)]
)
def test_nonlocal_or_bytes_paths_refuse(path: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        write_shot_data_file(
            data=torch.tensor([[True]]), path=path, format="b8", num_measurements=1
        )


def test_bool_copy_canonicalizes_true_storage(
    captured: tuple[Path, FakeWriter],
) -> None:
    path, backend = captured
    raw = bytearray([255])
    data = torch.frombuffer(raw, dtype=torch.bool).reshape(1, 1)
    write_shot_data_file(data=data, path=path, format="b8", num_measurements=1)
    assert raw == bytearray([255])
    assert backend.calls == [(b"\x01", 1, False, (1, 0, 0))]
    assert path.read_bytes() == b"\x01"


@pytest.mark.parametrize("value", [None, [[True]]])
def test_nontensor_refuses(value: torch.Tensor) -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        write_shot_data_file(data=value, path="unused", format="b8", num_measurements=1)


def test_invalid_or_incomplete_backend_modules_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(engine, "_stim", lambda: object())
    with pytest.raises(CapabilityError, match="invalid Stim SDK"):
        owner._backend()
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(owner, "import_module", lambda name: ModuleType("broken"))
    with pytest.raises(CapabilityError, match="embedded shot writer"):
        owner._backend()


def test_encoding_failure_retains_cause(captured: tuple[Path, FakeWriter]) -> None:
    path, backend = captured
    backend.fail = True
    with pytest.raises(OSError) as caught:
        write_shot_data_file(
            data=torch.tensor([[True]]), path=path, format="b8", num_measurements=1
        )
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert not path.exists()


def test_sdk_missing_numpy_refuses_without_staging_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(owner, "_backend", lambda: ModuleType("sdk"))
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(CapabilityError, match="NumPy"):
        write_shot_data_file(
            data=torch.tensor([[True]]),
            path=tmp_path / "new",
            format="b8",
            num_measurements=1,
        )
    assert list(tmp_path.iterdir()) == []
