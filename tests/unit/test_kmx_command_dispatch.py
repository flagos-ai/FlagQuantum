"""Exercise the whole original dispatch file and every real native command owner."""

import importlib
import json
import runpy
import sys
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.kmx_circuit import read_circuit_file
from flagquantum.ecosystem.kmx_command import main
from tests.unit.circuit_asset_fixtures import instructions

pytestmark = pytest.mark.unit

_ROUTES = {
    "circuit": "flagquantum.drawer.kmx_generate_command",
    "count_operations": "flagquantum.compiler.formats.kmx_operation_counts",
    "kmb2kmx": "flagquantum.compiler.formats.kmb_convert",
    "kmx2kmb": "flagquantum.compiler.formats.kmb_convert",
    "count_operations_sampled": "flagquantum.ecosystem.kmx_sampled_counts",
    "sample": "flagquantum.ecosystem.kmx_sample",
    "spin": "flagquantum.ecosystem.kmx_spin",
    "diagram": "flagquantum.drawer.kmx_diagram_command",
}


def observations() -> list[object]:
    data: object = json.loads(
        (Path(__file__).with_name("data") / "main_command.json").read_text()
    )
    if not isinstance(data, dict):
        raise ValueError("Invalid complete command source observations")
    commands: object = data.get("commands")
    if not isinstance(commands, list):
        raise ValueError("Invalid complete command source rows")
    return commands


@pytest.mark.parametrize("case_index", range(38))
def test_complete_original_dispatch(
    case_index: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observation = observations()[case_index]
    assert isinstance(observation, list)
    args, code, expected, error = observation
    assert isinstance(args, list)
    if args and args[0] in _ROUTES:
        name = args[0]
        owner = importlib.import_module(_ROUTES[name])

        def probe(argv: list[str] | None = None) -> int:
            assert argv is not None
            rest = argv
            if name in ("kmb2kmx", "kmx2kmb"):
                assert argv[0] == name
                rest = argv[1:]
            if rest and rest[0] == "throw":
                raise ValueError("command failure")
            print(name)
            for argument in rest:
                print(argument.encode().hex())
            return 7

        monkeypatch.setattr(owner, "main", probe)
    assert main(args) == code
    actual = capsys.readouterr()
    assert actual.out == expected
    assert actual.err == error.replace(
        "kickmix ", "python -m flagquantum.ecosystem.kmx_command "
    )


def test_c_argv_visibility_default_entry_and_module_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    owner = importlib.import_module(_ROUTES["sample"])
    received: list[list[str]] = []

    def probe(argv: list[str] | None = None) -> int:
        assert argv is not None
        received.append(argv)
        return 7

    monkeypatch.setattr(owner, "main", probe)
    monkeypatch.setattr(sys, "argv", ["flagquantum", "sample\0ignored", "é\0ignored"])
    assert main() == 7
    assert received == [["é"]]
    monkeypatch.delitem(sys.modules, "flagquantum.ecosystem.kmx_command")
    with pytest.raises(SystemExit) as result:
        runpy.run_module("flagquantum.ecosystem.kmx_command", run_name="__main__")
    assert result.value.code == 7
    assert received == [["é"], ["é"]]
    assert capsys.readouterr().out == ""


def test_all_real_command_owners_and_failure_output(tmp_path: Path) -> None:
    generated = tmp_path / "generated.kmx"
    assert (
        main(
            [
                "circuit",
                "--kind",
                "iadd",
                "--n",
                "3",
                "--out",
                str(generated),
                "--validate",
            ]
        )
        == 0
    )
    assert read_circuit_file(generated.read_bytes()).qubits > 0
    source = tmp_path / "simple.kmx"
    source.write_text('APPEND_TO_REGISTER q0 r0\nREGISTER r0 "out"\nX q0\n')
    binary, restored = tmp_path / "binary.kmb", tmp_path / "restored.kmx"
    assert main(["kmx2kmb", "--in", str(source), "--out", str(binary)]) == 0
    assert main(["kmb2kmx", "--in", str(binary), "--out", str(restored)]) == 0
    assert read_circuit_file(source.read_bytes()) == read_circuit_file(
        restored.read_bytes()
    )
    converted = read_circuit_file(restored.read_bytes())
    state = torch.tensor([0.3 + 0.4j, 0.5 + (0.5**0.5) * 1j], dtype=torch.complex128)
    actual = (
        Circuit(1, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(1, instructions(converted)))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, state.flip(0), atol=1e-12, rtol=0)
    for command, flags in (
        ("count_operations", []),
        ("diagram", ["--kind", "text"]),
        ("sample", ["--shots", "3", "--decimal"]),
        ("count_operations_sampled", ["--shots", "3"]),
        ("spin", ["--shots", "1", "--mode", "64"]),
    ):
        target = tmp_path / command
        assert main([command, "--in", str(binary), "--out", str(target), *flags]) == 0
        assert target.stat().st_size > 0
    protected = tmp_path / "protected"
    protected.write_bytes(b"caller data")
    assert (
        main(["diagram", "--in", str(tmp_path / "missing"), "--out", str(protected)])
        == 1
    )
    assert protected.read_bytes() == b"caller data"
