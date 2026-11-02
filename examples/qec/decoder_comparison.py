"""Compare complete decoder records on fixed independent toy DEMs.

Install ``pip install -e '.[pymatching]'``; CPU only, no Stim or sampling.
The models test decisions, not circuit construction or physical error rates.
"""

from flagquantum.qec import (
    BeliefPropagationOsdDecodeResult,
    DetectorErrorModel,
    MatchingDecodeResult,
    get_decoder,
)

CASES: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    (
        "unique_boundary",
        "error(0.125) D0 L0\n"
        "error(0.25) D1\n"
        "detector D0\n"
        "detector D1\n"
        "logical_observable L0\n",
        (0,),
    ),
    (
        "unique_triangle",
        "error(0.125) D0 L0\n"
        "error(0.125) D1\n"
        "error(0.125) D0 D1\n"
        "detector D0\n"
        "detector D1\n"
        "logical_observable L0\n",
        (0, 1),
    ),
    (
        "equal_cycle",
        "error(0.125) D0 D1 L0\n"
        "error(0.125) D1 D2\n"
        "error(0.125) D2 D3\n"
        "error(0.125) D0 D3\n"
        "detector D0\n"
        "detector D1\n"
        "detector D2\n"
        "detector D3\n"
        "logical_observable L0\n",
        (0, 2),
    ),
    (
        "near_cycle",
        "error(0.125) D0 D1 L0\n"
        "error(0.125) D1 D2\n"
        "error(0.125) D2 D3\n"
        "error(0.12500000000001) D0 D3\n"
        "detector D0\n"
        "detector D1\n"
        "detector D2\n"
        "detector D3\n"
        "logical_observable L0\n",
        (0, 2),
    ),
    (
        "objective_diamond",
        "error(0.125) D0 L0\n"
        "error(0.25) D0 D1\n"
        "error(0.25) D1\n"
        "error(0.25) D0 D2\n"
        "error(0.25) D2\n"
        "detector D0\n"
        "detector D1\n"
        "detector D2\n"
        "logical_observable L0\n",
        (0,),
    ),
)

NOTES = {
    "unique_boundary": "one explanation per syndrome",
    "unique_triangle": "unique minimum configuration; BP+OSD need not minimize",
    "equal_cycle": "opposite logical predictions can be exact minimum-weight ties",
    "near_cycle": "mathematical uniqueness and PyMatching precision are separate",
    "objective_diamond": "minimum configuration predicts 1; logical-class MAP predicts 0",
}


def _selected_indices(
    model: DetectorErrorModel,
    result: MatchingDecodeResult | BeliefPropagationOsdDecodeResult,
) -> tuple[int, ...]:
    if isinstance(result, BeliefPropagationOsdDecodeResult):
        return result.mechanisms
    indices = {
        (error.detectors, error.observables, error.probability): index
        for index, error in enumerate(model.errors)
    }
    return tuple(
        indices[
            (
                tuple(node for node in edge.detectors if node < model.num_detectors),
                edge.observables,
                edge.probability,
            )
        ]
        for edge in result.error_edges
    )


def _consistent(
    model: DetectorErrorModel,
    syndrome: tuple[int, ...],
    result: MatchingDecodeResult | BeliefPropagationOsdDecodeResult,
) -> bool:
    detectors = observables = 0
    for index in _selected_indices(model, result):
        error = model.errors[index]
        for detector in error.detectors:
            detectors ^= 1 << detector
        for observable in error.observables:
            observables ^= 1 << observable
    return detectors == sum(1 << bit for bit in syndrome) and observables == sum(
        1 << bit for bit in result.observables
    )


def main() -> None:
    """Print fifteen original records without requiring identical predictions."""
    for case, source, syndrome in CASES:
        print(f"{case}: {NOTES[case]}")
        model = DetectorErrorModel.from_stim_text(source)
        for name in (
            "minimum_weight_matching",
            "belief_propagation_osd",
            "pymatching",
        ):
            result = get_decoder(name, model).decode(syndrome)
            assert isinstance(
                result, (MatchingDecodeResult, BeliefPropagationOsdDecodeResult)
            )
            consistent = _consistent(model, syndrome, result)
            assert consistent
            print(
                f"{case} {name} syndrome={syndrome} consistent={consistent} {result!r}"
            )


if __name__ == "__main__":
    main()
