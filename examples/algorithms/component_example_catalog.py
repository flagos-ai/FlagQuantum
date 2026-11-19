"""Construct a documented example and execute it through circuit.append(op)."""

import torch

import flagquantum as fq
from examples.algorithms.register_component_tutorial import WordSwap
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    component_example,
)


@component_example
def _word_swap() -> WordSwap:
    """Exchange two three-bit words using nine controlled-NOT gates."""
    return WordSwap(3)


SWAP_DOCUMENTATION = ComponentDocumentation(
    component_type=WordSwap,
    examples=(_word_swap,),
    import_line="from examples.algorithms.register_component_tutorial import WordSwap",
)


def main() -> None:
    example = SWAP_DOCUMENTATION.call_graph_example
    assert example is not None
    operation = example.make()
    coefficients = torch.arange(128, dtype=torch.float64)
    state = torch.complex(coefficients + 1, coefficients.flip(0) + 5)
    state /= state.norm()
    circuit = fq.Circuit(7, inputs=state, dtype=torch.complex128)
    circuit.append(operation.make_on((0, 1, 2), (3, 4, 5)))
    expected = state.reshape(8, 8, 2).transpose(0, 1).flatten()
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=0)
    assert len(circuit.to_ir().instructions) == 9
    print("Documented three-bit swap: nine CX gates, exact complex amplitudes.")


if __name__ == "__main__":
    main()
