# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 basic_gates/cnot.py and cnot.ipynb.

"""Native CNOT teaching views over the existing leaf, truth table and drawer."""

from examples.algorithms.classical_tensor_crosscheck import (
    crosscheck_program,
    crosscheck_values,
)
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import MusicalScoreData, register_score_data
from flagquantum.simulation.classical_truth_table import (
    format_classical_truth_table,
    register_classical_truth_table,
)

CNOT = CliffordCall("cx", 2)


def cnot_signature() -> Signature:
    """Use the same signature as the stored primitive body."""
    return Signature(crosscheck_program("cnot").functions[0].registers or ())


def cnot_graph() -> SymbolicCallGraph[CliffordCall]:
    """One atomic node, with neither outgoing edges nor a decomposition."""
    return symbolic_call_graph(CNOT, {}, max_depth=1)


def cnot_truth_table() -> str:
    """Enumerate valid words using the shared value and truth-table owners."""
    signature = cnot_signature()
    return format_classical_truth_table(
        *register_classical_truth_table(
            signature,
            lambda values: tuple(crosscheck_values("cnot", values).values()),
        )
    )


def cnot_score_data() -> MusicalScoreData:
    """Draw the original filled control and plus target in register order."""
    names = tuple(register.name for register in cnot_signature())
    return register_score_data(
        layout_register_score(
            dict.fromkeys(names, 1), names, (RegisterStep("CNOT", names, names, "cx"),)
        )
    )
