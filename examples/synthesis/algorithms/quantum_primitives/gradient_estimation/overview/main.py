"""Run every source gradient experiment,sweep and actual readout locally."""

from pathlib import Path

from flagquantum import synth

from .ansatz import auxiliary_phase, build_program
from .plots import gradient_plot, phase_plot, sweep_plot
from .workflow import (
    L_VALUES,
    M_VALUES,
    Model,
    build_two_dimensional,
    phase_table,
    sample,
    success_rate,
    verify_auxiliary,
    verify_program,
    verify_two_dimensional,
)


def run_example() -> None:
    directory = Path("artifacts/development/gradient_estimation")
    auxiliary = synth.synthesize(auxiliary_phase, max_workspace=0)
    print("Auxiliary:", verify_auxiliary(auxiliary), auxiliary.report["resources"])
    model = Model(0, 0.5, 0.25)
    print(
        phase_plot(model, phase_table(auxiliary, model), directory, "auxiliary_phase")
    )
    for name, model in (
        ("linear_phase", Model(0, 0.5, 0.25)),
        ("quadratic_phase_large", Model(0.5, 0.25, 0.1, window=2)),
        ("quadratic_phase_small", Model(0.5, 0.25, 0.1, window=0.2)),
    ):
        p = build_program(model.coefficients, inverse_fourier=False)
        print(
            name, verify_program(p, model, inverse_fourier=False), p.report["resources"]
        )
        print(phase_plot(model, phase_table(p, model), directory, name))
    for index, (name, model) in enumerate(
        (
            ("negative_linear", Model(0, -0.5, 0.25)),
            ("noninteger_linear", Model(0, 0.55, 0.25)),
            ("quadratic_small", Model(0.6, 0.25, 0.1, window=0.1)),
            ("quadratic_large", Model(0.6, 0.25, 0.1, window=1)),
        )
    ):
        p = build_program(model.coefficients)
        print(
            name, verify_program(p, model, inverse_fourier=True), p.report["resources"]
        )
        observations = sample(p, seed=3000 + index)
        outcome = success_rate(observations, model, (model.linear,))
        print(observations, outcome)
        if name == "quadratic_small" and outcome.rate <= 0.9:
            raise AssertionError("source small-window success threshold failed")
        if name == "quadratic_large" and outcome.rate >= 0.5:
            raise AssertionError("source large-window success threshold failed")
        print(gradient_plot(model, observations, directory, name))
    for parameter, values in (("m", M_VALUES), ("l", L_VALUES)):
        rates = []
        for index, value in enumerate(values):
            model = (
                Model(0.6, 0.25, 0.1, window=0.1, m=value)
                if parameter == "m"
                else Model(0.6, 0.25, 0.1, window=value, m=1)
            )
            p = build_program(model.coefficients)
            print(verify_program(p, model, inverse_fourier=True))
            observations = sample(
                p, seed=3100 + index + (100 if parameter == "l" else 0)
            )
            result = success_rate(
                observations,
                model,
                (0.25,),
                tolerance=0.2 if parameter == "m" else None,
            )
            rates.append(result.rate)
            print(parameter, value, result)
        print(sweep_plot(values, tuple(rates), parameter, directory))
    p = build_two_dimensional()
    print("Two-dimensional:", verify_two_dimensional(p), p.report["resources"])
    observations = sample(p, seed=3400)
    result = success_rate(observations, Model(0.6, 0.25, 0.1, window=0.1), (0.25, -0.5))
    print(observations, result)
    if result.rate <= 0.9:
        raise AssertionError("source two-dimensional success threshold failed")


if __name__ == "__main__":
    run_example()
