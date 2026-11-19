"""Compose fresh source embedding, coherent validation and bounded native VQE."""

from dataclasses import dataclass

from . import model, reference, variational


@dataclass(frozen=True)
class Result:
    molecule: model.Model
    optimization: variational.Optimization
    state_errors: tuple[float, ...]
    energy_errors: tuple[float, ...]


def run(prepared: model.Model | None = None) -> Result:
    molecule = model.regenerate() if prepared is None else prepared
    generators = variational.operators(molecule)
    errors = variational.check_states(molecule, generators)
    energy_errors = reference.check_energies(molecule, generators)
    optimization = variational.optimize(molecule, generators)
    return Result(molecule, optimization, errors, energy_errors)
