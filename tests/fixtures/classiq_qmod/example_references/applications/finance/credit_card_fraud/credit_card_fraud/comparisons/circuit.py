"""Test-only unchanged source against the independent native Circuit reference."""

from pathlib import Path

from examples.synthesis.applications.finance.credit_card_fraud.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.finance.credit_card_fraud.main import DATA_PAIRS
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    for data1, data2 in DATA_PAIRS:
        program = synth.synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"data1": data1, "data2": data2},
            max_workspace=0,
        )
        compare_states(program, reference_circuit(data1, data2), ("qba",))


if __name__ == "__main__":
    run_example()
