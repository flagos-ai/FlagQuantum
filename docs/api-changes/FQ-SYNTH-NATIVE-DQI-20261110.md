# Native DQI symmetric-weight preparation and coherent syndrome decoding

Status: accepted locally after 61 acceptance/related tests, 29 catalog/API tests,
strict typing of five Python files and final fresh 216-original plus approved-six
replay. All 29 production pins and original resource records are unchanged.

Native default: 15 physical bits / 3 work bits / 403 operations / depth 352;
original: 14 / 2 / 3,802 / 1,988. Complete default amplitude error is below
3.6e-16; clean-work probability is zero. All three weight vectors and coherent
nonzero decoder XOR/inverse cases pass. Native coverage is 200/216; Notebook
coverage remains 24/213.

The frozen DQI program prepares a superposition of weight-zero, weight-one and
weight-two Dicke states on six error bits, adds the retained parity phase,
computes a six-by-six GF(2) transform into six syndrome bits, XORs a 22-entry
syndrome decoder into the error word and Fourier-transforms the syndrome word.
Keep the complete quantum computation; do not substitute a precomputed final
solution state.

Existing real-amplitude preparation can directly prepare the symmetric-weight
state: each basis word of weight k has amplitude weight[k]/sqrt(comb(6,k)).
This replaces the source's consumed binary/unary preparation storage without
retaining or freeing those intermediate ports. The frozen fourth weight is zero;
this example accepts normalized finite real weights with that same support,
not a general all-weight Dicke preparation claim. Preserve all model columns,
lookup entries, parity signs, both six-bit output words and absolute phase.

The native GF(2) and decoder helpers borrow disjoint existing storage. Decoder
pattern masks are static classical data; controls, XOR and inverse masks execute
coherently. Allocation, real preparation, CX, raw array control, X, phase,
Hadamard and checked adjoint reuse existing public APIs. No production Compiler,
Runtime, protected export or API signature change is required.

Acceptance must compare all 4,096 logical amplitudes and every clean work
amplitude against an independent Circuit for the original weights and two
additional normalized nonzero signed weight vectors. Compare the original gate
program as well, preserving its checked zero-free behavior in test fixtures.
Separately exercise the decoder on coherent syndrome and nonzero complex error
states to prove XOR semantics, restore it by checked inverse, reject aliases and
invalid weights, and check all resource budgets. Preserve original resource
records and report any native overhead honestly.

After final full216 plus approved6 replay, archive immutable evidence and update
catalog and delivery ledger. Native coverage starts at199/216; Notebook coverage
remains24/213. Classification is `single_device_fast_path`. No integration PR,
push or full remote CI before216 native quantum-program acceptances.
