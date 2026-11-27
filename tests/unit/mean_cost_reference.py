"""Source-owned static encoder models for independent lifetime comparisons."""

from typing import Any

from tests._reference_dependencies import unavailable_reference


def source_encoder(forward_peak: int, inverse_peak: int) -> Any:
    from functools import cached_property

    import attrs

    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran._infra.data_types import BQUInt, QAny
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
    from qualtran.resource_counting._bloq_counts import GateCounts, QECGatesCost
    from qualtran.resource_counting._qubit_counts import QubitCount

    def control_registers(self: Any) -> tuple[Any, ...]:
        return ()

    def selection_registers(self: Any) -> tuple[Any, ...]:
        return (SourceRegister("q", BQUInt(1, 2)),)

    def target_registers(self: Any) -> tuple[Any, ...]:
        return (SourceRegister("t", QAny(1)),)

    def decompose(self: Any, context: Any, **quregs: Any) -> Any:
        yield cirq.CNOT(quregs["q"][0], quregs["t"][0])

    def adjoint(self: Any) -> Any:
        return attrs.evolve(self, reverse=not self.reverse)

    def static_costs(self: Any, cost_key: Any) -> Any:
        if isinstance(cost_key, QubitCount):
            return inverse_peak if self.reverse else forward_peak
        if isinstance(cost_key, QECGatesCost):
            return GateCounts(
                t=5 if self.reverse else 3, clifford=8 if self.reverse else 2
            )
        return NotImplemented

    encoder_type = attrs.make_class(
        "Encoder",
        {"reverse": attrs.field(default=False)},
        bases=(SelectOracle,),
        class_body={
            "__module__": __name__,
            "control_registers": property(control_registers),
            "selection_registers": cached_property(selection_registers),
            "target_registers": cached_property(target_registers),
            "decompose_from_registers": decompose,
            "adjoint": adjoint,
            "my_static_costs": static_costs,
        },
        frozen=True,
    )
    return encoder_type()
