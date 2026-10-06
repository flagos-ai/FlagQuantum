"""Compare the gradient methods one circuit can use, and read what each reported.

Examples
--------
Run every method and print the method, exactness, step, and error::

    python examples/gradient_methods/run.py
"""

from __future__ import annotations

import torch

import flagquantum as fq

PARAMETERS = torch.tensor([0.3, 0.7, -0.4], dtype=torch.float64)


def build_circuit(parameters: torch.Tensor) -> fq.Circuit:
    """One parameterized circuit, independent of how it is executed."""
    return (
        fq.Circuit(2)
        .ry(0, theta=parameters[0])
        .rx(0, theta=parameters[1])
        .ry(1, theta=parameters[2])
        .cx(1, 0)
    )


def loss(circuit: fq.Circuit) -> torch.Tensor:
    """The scalar the derivative is taken of."""
    result = fq.run(
        circuit,
        outputs=fq.expectation(fq.Z(0)),
    )
    return result.expectations[0]


def relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    scale = float(reference.abs().max())
    return float((candidate - reference).abs().max()) / scale


def two_expectations(parameters: torch.Tensor) -> torch.Tensor:
    """A program returning several values, so no single scalar gradient exists."""
    result = fq.run(
        build_circuit(parameters),
        outputs=[fq.expectation(fq.Z(0)), fq.expectation(fq.Z(1))],
    )
    return torch.stack(list(result.expectations))


def main() -> None:
    reference = fq.gradient(build_circuit, PARAMETERS, loss)
    print(f"{reference.method:<18} exact={reference.exact} step={reference.step}")

    # The same circuit in another storage mode is the same derivative.
    mps = fq.gradient(
        build_circuit,
        PARAMETERS,
        lambda circuit: fq.run(
            circuit,
            options=fq.ExecutionOptions(mode="mps"),
            outputs=fq.expectation(fq.Z(0)),
        ).expectations[0],
    )
    print(
        f"{'mps/autograd':<18} exact={mps.exact} error={relative_error(mps.gradient, reference.gradient):.3e}"
    )

    shifted = fq.gradient(build_circuit, PARAMETERS, loss, method="parameter_shift")
    print(
        f"{shifted.method:<18} exact={shifted.exact} error={relative_error(shifted.gradient, reference.gradient):.3e}"
    )

    for method in ("finite_difference", "spsa"):
        # directions and generator belong to spsa alone, so they are passed there.
        controls = (
            {"directions": 512, "generator": torch.Generator().manual_seed(0)}
            if method == "spsa"
            else {}
        )
        result = fq.gradient(build_circuit, PARAMETERS, loss, method=method, **controls)
        print(
            f"{result.method:<18} exact={result.exact} "
            f"step={result.step:.3e} "
            f"error={relative_error(result.gradient, reference.gradient):.3e}"
        )

    # A program whose scoring step runs outside the autograd graph still gets a
    # derivative, and `auto` reports that it had to fall back to a difference.
    def detached(parameters: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return loss(build_circuit(parameters))

    fell_back = fq.gradient(detached, PARAMETERS)
    print(
        f"{fell_back.method:<18} exact={fell_back.exact} "
        f"step={fell_back.step:.3e} "
        f"error={relative_error(fell_back.gradient, reference.gradient):.3e}"
    )

    # Two outputs at once. There is no single scalar gradient, so ask for the
    # Jacobian, then apply it along one parameter direction and one output
    # direction. The two products must be adjoint: <Jv, c> == <v, J^T c>.
    jacobian = fq.jacobian(two_expectations, PARAMETERS)
    tangent = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    cotangent = torch.tensor([0.25, -1.5], dtype=torch.float64).reshape(2, 1)
    forward = fq.jvp(two_expectations, PARAMETERS, tangent)
    reverse = fq.vjp(two_expectations, PARAMETERS, cotangent)
    print(
        f"{'jacobian':<18} shape={tuple(jacobian.shape)} "
        f"matvec_error="
        f"{relative_error(forward.reshape(-1), jacobian.reshape(-1, 3) @ tangent):.3e}"
    )
    # This circuit is complex64 while the parameters are float64, so the
    # residual of the adjoint identity is the amplitude dtype's epsilon.
    print(
        f"{'jvp/vjp':<18} adjoint_error="
        f"{abs(float((forward * cotangent).sum()) - float((reverse * tangent).sum())):.3e}"
    )


if __name__ == "__main__":
    main()
