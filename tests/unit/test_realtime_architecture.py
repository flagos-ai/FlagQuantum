"""Reject dependencies that would turn messaging into an execution engine."""

from pathlib import Path

import pytest

import tools.check_architecture as checker

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "source",
    (
        "import flagquantum.qec",
        "from flagquantum import runtime",
        "from .. import remote",
        "from ..compiler import CouplingMap",
        "import torch",
        "import numpy",
    ),
)
def test_realtime_rejects_external_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    package = tmp_path / "flagquantum"
    monkeypatch.setattr(checker, "PACKAGE", package)
    path = package / "realtime" / "consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert any(
        "realtime may import only itself" in error
        for error in checker._realtime_dependency_errors(path)
    )


@pytest.mark.parametrize(
    "domain", checker.CONFIG["realtime_boundaries"]["forbidden_consumers"]
)
@pytest.mark.parametrize(
    "source",
    (
        "import flagquantum.realtime",
        "from flagquantum import realtime",
        "from .. import realtime",
        "from ..realtime.protocol import RPCHeader",
        "from flagquantum.remote import realtime",
        "from ..remote.realtime import RPCHeader",
    ),
)
def test_execution_layers_cannot_acquire_a_realtime_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: str, source: str
) -> None:
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    package = tmp_path / "flagquantum"
    monkeypatch.setattr(checker, "PACKAGE", package)
    path = package / domain / "consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert any(
        "imports realtime before integration admission" in error
        for error in checker._realtime_dependency_errors(path)
    )


@pytest.mark.parametrize(
    "source",
    (
        "from .protocol import RPCHeader",
        "import flagquantum.realtime.protocol",
        "from typing import Protocol",
        "import struct",
    ),
)
def test_realtime_allows_its_own_protocol_and_standard_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    package = tmp_path / "flagquantum"
    monkeypatch.setattr(checker, "PACKAGE", package)
    path = package / "realtime" / "consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert checker._realtime_dependency_errors(path) == ()
