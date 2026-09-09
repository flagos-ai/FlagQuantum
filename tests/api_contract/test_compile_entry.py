"""User-facing compiler selection stays explicit and fail-closed."""

import flagquantum as fq


def test_fq_compile_uses_builtin_compiler_by_default():
    source = fq.Circuit(2).h(0).cx(0, 1)

    compiled = fq.compile(source)

    assert isinstance(compiled, fq.CircuitIR)
    assert compiled.n_wires == 2


def test_fq_compile_resolves_quafu_target_for_named_plugin(monkeypatch):
    source = fq.Circuit(2).h(0).cx(0, 1)
    chip_info = {
        "calibration_time": "2026-09-09 12:00:00",
        "qubits_info": {"Q3": {}, "Q4": {}},
        "couplers_info": {"C0": {"qubits_index": [3, 4], "fidelity": 0.99}},
    }
    captured = {}

    class Provider:
        def fetch_chip_info(self, backend):
            assert backend == "ScQ-P10"
            return chip_info

    def compile_with_extension(program, *, extension, target):
        captured.update(extension=extension, target=target)
        return program

    monkeypatch.setattr("flagquantum.remote.QuafuProvider", Provider)
    monkeypatch.setattr(
        "flagquantum.ecosystem.extensions.compile_with_extension",
        compile_with_extension,
    )

    compiled = fq.compile(source, compiler="qsteed", target="quafu:ScQ-P10")

    assert compiled == source.to_ir()
    assert captured == {
        "extension": "qsteed",
        "target": {
            "provider": "quafu",
            "backend": "ScQ-P10",
            "chip_info": chip_info,
        },
    }
