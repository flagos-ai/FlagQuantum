"""Run the existing QEC workflow protected by QEC-RESTRUCTURE-001."""

from flagquantum.qec import (
    AUTHORITY_NAME,
    DetectorErrorModel,
    ErrorEvent,
    ErrorSchedule,
    get_decoder,
    run_repetition_memory_experiment,
)


def main() -> None:
    model = DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
    )
    decoder = get_decoder(AUTHORITY_NAME, model)
    print("Correction observables:", decoder.decode((0,)).observables)
    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(1, 1),)),
        rounds=3,
        shots=2,
        seed=53,
        strategy="trajectory",
        feedback_mode="offline_pauli_frame",
    )
    print("Logical error rate:", result.logical_error_rate)
    print("Decoded data:", result.shot_records[0].decoded_data_bits)


if __name__ == "__main__":
    main()
