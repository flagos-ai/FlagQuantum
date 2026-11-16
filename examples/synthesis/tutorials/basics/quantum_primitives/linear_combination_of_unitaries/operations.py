"""Reversible coherent controller loading."""

from flagquantum.synth import QNum, inplace_prepare_state, qfunc


@qfunc
def prepare(probabilities: list[float], controller: QNum) -> None:
    inplace_prepare_state(probabilities, 0, controller)
