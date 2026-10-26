"""Complete sparse-shot records; install with ``pip install -e '.[pymatching]'``."""

from flagquantum.qec import DetectorErrorModel, get_decoder
from flagquantum.qec.decoders.batch import decode_batch


def main() -> None:
    model = DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
    )
    shots = ((), (0,), (0,))
    for name in (
        "minimum_weight_matching",
        "sliding_window_matching",
        "belief_propagation",
        "belief_propagation_osd",
        "pymatching",
    ):
        decoder = get_decoder(name, model)
        results = decode_batch(decoder, shots)
        assert results == tuple(decoder.decode(shot) for shot in shots)
        print(name, tuple(result.observables for result in results))


if __name__ == "__main__":
    main()
