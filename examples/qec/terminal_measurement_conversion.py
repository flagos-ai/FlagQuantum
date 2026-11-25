"""Compare ordinary simulation and terminal QEC readout on one circuit."""

import math

import torch

import flagquantum as fq
from flagquantum.compiler import Layout, apply_layout
from flagquantum.core.ir import CircuitIR
from flagquantum.qec.execution.context import MeasurementMap
from flagquantum.qec.execution.measurement_conversion import compile_m2d_converter


def main() -> None:
    theta = fq.Parameter("theta")
    circuit = (
        fq.Circuit(3, dtype=torch.complex128).h(0).cx(0, 1).rx(2, theta=theta).x(1).x(1)
    )

    bound = circuit.bind_parameters({theta: math.pi})
    compiled = fq.compile(bound)
    layout = Layout((2, 0, 1))
    placed = apply_layout(compiled, layout)
    options = fq.ExecutionOptions(mode="auto", device="cpu", allow_approximate=False)

    def probabilities(program: fq.Circuit | CircuitIR) -> torch.Tensor:
        result = fq.run(program, options=options, outputs=fq.probabilities())
        value = result.probabilities
        assert isinstance(value, torch.Tensor)
        return value

    def z(program: fq.Circuit | CircuitIR, qubit: int) -> float:
        result = fq.run(program, options=options, outputs=fq.expectation(fq.Z(qubit)))
        return float(result.expectation().item())

    torch.testing.assert_close(
        probabilities(bound), probabilities(compiled), atol=1e-12, rtol=0
    )
    assert math.isclose(z(bound, 2), -1, abs_tol=1e-12, rel_tol=0)
    assert math.isclose(z(compiled, 2), -1, abs_tol=1e-12, rel_tol=0)
    assert math.isclose(z(placed, 1), -1, abs_tol=1e-12, rel_tol=0)
    assert math.isclose(
        z(circuit.bind_parameters({theta: 0.0}), 2), 1, abs_tol=1e-12, rel_tol=0
    )

    readout = (2, 0, 1)
    detector_map = MeasurementMap(3, ((1, 2),))
    observable_map = MeasurementMap(3, ((0,),))
    raw = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.bool)

    for program in (bound, compiled, placed):
        converter = compile_m2d_converter(
            program,
            measurement_qubits=readout,
            detector_map=detector_map,
            observable_map=observable_map,
        )
        assert converter.measurement_qubits == readout
        dets, obs = converter.convert(measurements=raw, separate_observables=True)
        assert dets.tolist() == [[False], [True], [False]]
        assert obs.tolist() == [[False], [False], [True]]

    print("binding, terminal readout and QEC events agree")


if __name__ == "__main__":
    main()
