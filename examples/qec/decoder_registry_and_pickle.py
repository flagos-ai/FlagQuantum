"""Use the existing decoder registry and retain historical pickle paths."""

import pickle

from flagquantum.qec import DetectorErrorModel, decoder_names, get_decoder

model = DetectorErrorModel.from_stim_text(
    "error(0.125) D0 L0\nerror(0.25) D0 D1\n"
    "detector D0\ndetector D1\nlogical_observable L0\n"
)
for name in (
    "minimum_weight_matching",
    "belief_propagation",
    "belief_propagation_osd",
):
    decoder = get_decoder(name, model)
    print(name, decoder.decode((0,)).observables)
decoder = get_decoder("minimum_weight_matching", model)
restored = pickle.loads(pickle.dumps(decoder))
print("Historical pickle class:", type(restored).__module__)
print("Same class:", type(restored) is type(decoder))
print("Registered decoders:", len(decoder_names()))
