"""Source export, resource ownership and bounded optional rendering."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.drawer import LatexFigure, draw, draw_hybrid, draw_latex
from flagquantum.drawer import latex_figure as rendering

pytestmark = pytest.mark.unit


def test_lppl_runtime_and_license_are_distributed_verbatim(tmp_path: Path) -> None:
    source = draw_latex(fq.Circuit(1).h(0)).export_bundle(tmp_path)
    root = source.parent
    paths = sorted(
        [
            *root.glob("yquant*.sty"),
            *root.glob("yquant*.tex"),
            root / "licenses/yquant.txt",
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(
            path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes()
        )
    assert digest.hexdigest() == (
        "644ff1c188a322de1b0abf9eb00446230653ca74f94a92faa0b70a38584d8a7f"
    )


def test_circuit_ir_json_and_fragment_export_agree() -> None:
    theta = fq.Parameter("theta_0")
    circuit = fq.Circuit(3).h(0).cx(2, 0).ccx(2, 0, 1).ry(1, 2 * theta)
    figure = circuit.draw(format="latex")
    assert isinstance(figure, LatexFigure)
    assert figure.source == draw_latex(circuit.to_ir()).source
    assert figure.source == draw_latex(circuit.to_ir().to_dict()).source
    assert "cnot q[0] | q[2];" in figure.source
    assert "cnot q[1] | q[2], q[0];" in figure.source
    assert r"R_y((2\cdot \theta_{0}))" in figure.source
    fragment = draw_latex(circuit, standalone=False)
    assert r"\documentclass" not in fragment.source
    assert r"\begin{yquant}" in fragment.source
    assert str(figure) == figure.source


def test_readout_semantics_and_parameter_gradient_ownership() -> None:
    parameter = torch.tensor(0.23, requires_grad=True)
    ir = CircuitIR(
        2,
        (Instruction("ry", (0,), {"theta": parameter}),),
        observables=(ObservableNode("z", (0,), coefficient=0.5),),
        measurements=(MeasurementNode("probabilities", (0, 1), shots=512),),
    )
    source = draw_latex(ir).source
    assert "Observable: z, coefficient=0.5" in source
    assert "Readout: probabilities, shots=512" in source
    assert "measure q" not in source
    assert ir.instructions[0].params["theta"] is parameter
    assert parameter.requires_grad and parameter.grad is None
    (parameter * 3).backward()
    torch.testing.assert_close(parameter.grad, torch.tensor(3.0))


def test_parameter_order_and_tex_escaping() -> None:
    circuit = (
        fq.Circuit(2)
        .u2(0, phi=0.1, lbd=0.2)
        .crz(0, 1, theta=fq.Parameter(r"x%\input{secret}"))
    )
    source = draw_latex(circuit).source
    assert "U_2(0.1,0.2)" in source
    assert r"\%\textbackslash{}input\{secret\}" in source
    assert r"\input{secret}" not in source
    assert "q[1] | q[0];" in source


@pytest.mark.parametrize(
    "instruction",
    [
        Instruction("bit_flip", (0,), {"probability": 0.2}),
        Instruction("h", (0,), metadata={"condition": 0}),
        Instruction("h", (0,), metadata={"is_dynamic": True}),
        Instruction("custom", (0,), matrix=torch.eye(2)),
    ],
)
def test_semantics_without_a_rendering_rule_are_rejected(
    instruction: Instruction,
) -> None:
    with pytest.raises(ValueError, match="does not support"):
        draw_latex(CircuitIR(1, (instruction,)))


def test_batched_parameters_are_not_silently_summarized() -> None:
    with pytest.raises(ValueError, match="scalar gate parameters"):
        draw_latex(fq.Circuit(1).ry(0, torch.tensor([0.1, 0.2])))


def test_portable_bundle_and_conflict_preflight(tmp_path: Path) -> None:
    figure = draw_latex(fq.Circuit(1).h(0), standalone=False)
    root = tmp_path / "bundle"
    source = figure.export_bundle(root)
    assert r"\documentclass" in source.read_text()
    for name in (
        "flaghybrid.sty",
        "yquant.sty",
        "yquant-lang.tex",
        "Box.sty",
        "licenses/yquant.txt",
        "licenses/PlotNeuralNet.txt",
    ):
        assert (root / name).is_file()
    assert figure.export_bundle(root) == source
    (root / "Box.sty").write_text("caller-owned")
    source.unlink()
    with pytest.raises(FileExistsError, match="Box.sty"):
        figure.export_bundle(root)
    assert not source.exists()
    assert (root / "Box.sty").read_text() == "caller-owned"
    other = tmp_path / "ancestor-conflict"
    other.mkdir()
    (other / "licenses").write_text("caller-owned directory name")
    with pytest.raises(FileExistsError, match="non-directory"):
        figure.export_bundle(other)
    assert not (other / "figure.tex").exists()
    assert figure.save(tmp_path / "source.tex").read_text() == figure.source


def test_hybrid_source_is_explicit_and_copies_labels() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    figure = draw_hybrid(
        circuit, encoder="CNN_1%", encoder_features=3, head_outputs=2, training=True
    )
    assert r"CNN\_1\%" in figure.source
    assert "{3 features}" in figure.source and "{2 outputs}" in figure.source
    assert "cnot q[1] | q[0];" in figure.source
    assert "predictions" in figure.source and "head update" in figure.source
    assert "Loss + optimizer" not in draw_hybrid(circuit).source
    for count in (True, 0, -1, 1.5):
        with pytest.raises((TypeError, ValueError)):
            draw_hybrid(circuit, head_outputs=count)


def test_missing_tools_and_plain_text_repr_do_not_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rendering.shutil, "which", lambda name: None)
    figure = draw_latex(fq.Circuit(1).h(0))
    assert figure._repr_mimebundle_(include=["text/plain"]) == {
        "text/plain": repr(figure)
    }
    with pytest.raises(ImportError, match="source export needs neither"):
        figure.render()
    with pytest.raises(ImportError, match="SVG conversion"):
        figure.render("svg")
    with pytest.raises(ImportError, match="SVG conversion"):
        figure._repr_mimebundle_(include=["image/svg+xml"])


def test_render_cache_and_mime_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    directories: list[Path] = []
    monkeypatch.setattr(rendering.shutil, "which", lambda name: "/tools/" + name)

    def run(
        command: list[str], *, cwd: Path, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 120
        calls.append(command)
        directories.append(cwd)
        if command[0].endswith("tectonic"):
            assert "--untrusted" in command
            assert (cwd / "yquant.sty").exists()
            (cwd / "figure.pdf").write_bytes(b"%PDF-1.7\n")
        elif "-svg" in command:
            (cwd / "figure.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        else:
            (cwd / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(rendering.subprocess, "run", run)
    figure = draw_latex(fq.Circuit(1).h(0))
    assert "image/svg+xml" in figure._repr_mimebundle_()
    assert figure.render("svg") == figure.render("svg")
    figure.save(tmp_path / "test.pdf")
    assert "image/png" in figure._repr_mimebundle_(include=["image/png"])
    assert len(calls) == 3  # One compilation; SVG and PNG each converted once.
    assert not any(path.exists() for path in directories)


@pytest.mark.parametrize("failure", ["compile", "timeout", "converter"])
def test_render_failures_keep_diagnostics_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    directories: list[Path] = []
    monkeypatch.setattr(rendering.shutil, "which", lambda name: "/tools/" + name)

    def run(
        command: list[str], *, cwd: Path, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        directories.append(cwd)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "compile" or "-svg" in command:
            raise subprocess.CalledProcessError(
                1, command, output="diagnostic: unsupported command"
            )
        (cwd / "figure.pdf").write_bytes(b"%PDF-1.7\n")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(rendering.subprocess, "run", run)
    figure = draw_latex(fq.Circuit(1).h(0))
    with pytest.raises(
        RuntimeError, match="timed out" if failure == "timeout" else "diagnostic"
    ):
        figure.render("svg")
    assert not any(path.exists() for path in directories)


def test_display_option_invokes_ipython_without_changing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("IPython")
    import IPython.display

    seen: list[object] = []
    monkeypatch.setattr(IPython.display, "display", seen.append)
    figure = fq.Circuit(1).h(0).draw(format="latex", display=True)
    assert seen == [figure]
    hybrid = draw_hybrid(fq.Circuit(1), display=True)
    assert seen == [figure, hybrid]
    with pytest.raises(TypeError, match="training must be"):
        draw_hybrid(fq.Circuit(1), training=1)
    assert isinstance(draw(fq.Circuit(1).h(0)), str)


def test_additional_native_scalar_types_and_invalid_programs() -> None:
    numpy = pytest.importorskip("numpy")
    circuit = fq.Circuit(1).ry(0, numpy.float32(0.25))
    assert "R_y(0.25)" in draw_latex(circuit).source
    for program in (
        object(),
        "not a circuit",
        {"version": "9.0", "n_wires": 1, "instructions": []},
    ):
        with pytest.raises((TypeError, ValueError)):
            draw_latex(program)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_render_budget_is_rejected_without_compilation(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        LatexFigure("source", timeout=timeout)


def test_tex_live_disables_shell_escape_and_checks_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rendering.shutil, "which", lambda name: "/tools/" + name)

    def run(
        command: list[str], *, cwd: Path, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert "-no-shell-escape" in command
        assert "-halt-on-error" in command
        (cwd / "figure.pdf").write_bytes(b"%PDF-1.7\n")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(rendering.subprocess, "run", run)
    figure = draw_latex(fq.Circuit(1).h(0), engine="pdflatex")
    assert figure.render().startswith(b"%PDF-")
    with pytest.raises(ValueError, match="engine must be"):
        draw_latex(fq.Circuit(1), engine="unknown-engine").render()


@pytest.mark.parametrize("failure", ["missing", "corrupt", "oserror"])
def test_bad_external_compiler_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(rendering.shutil, "which", lambda name: "/tools/" + name)

    def run(
        command: list[str], *, cwd: Path, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if failure == "oserror":
            raise OSError("executable cannot run")
        if failure == "corrupt":
            (cwd / "figure.pdf").write_bytes(b"invalid PDF")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(rendering.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        draw_latex(fq.Circuit(1).h(0)).render()


@pytest.mark.parametrize(
    "options",
    [
        {"source": 1},
        {"source": "text", "standalone": "yes"},
        {"source": "text", "engine": 1},
    ],
)
def test_invalid_figure_configuration_is_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        LatexFigure(**options)


def test_unsupported_image_extensions_do_not_start_rendering(tmp_path: Path) -> None:
    figure = draw_latex(fq.Circuit(1))
    with pytest.raises(ValueError, match="render format"):
        figure.render("gif")
    with pytest.raises(ValueError, match="output extension"):
        figure.save(tmp_path / "figure.gif")
    assert not (tmp_path / "figure.gif").exists()


@pytest.mark.parametrize("format", ["png", "svg"])
@pytest.mark.parametrize("data", [None, b"invalid image"])
def test_missing_or_corrupt_converter_images_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    format: str,
    data: bytes | None,
) -> None:
    monkeypatch.setattr(rendering.shutil, "which", lambda name: "/tools/" + name)
    monkeypatch.setattr(LatexFigure, "_compile_pdf", lambda self: b"%PDF-1.7\n")

    def run(
        command: list[str], *, cwd: Path, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if data is not None:
            (cwd / ("figure." + format)).write_bytes(data)
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(rendering.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        draw_latex(fq.Circuit(1)).render(format)
