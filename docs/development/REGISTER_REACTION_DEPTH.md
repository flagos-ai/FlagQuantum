# Register reaction depth

`compiler/artifacts/reaction_depth.py` follows locked Kickmix `circuit_util.cc`
and its header (K-0021/K-0022, blobs
`256afad75334d9d31864aa4559d3543fb951fa2b` and
`0cc9293c5813af2f0572e516e3c83acbfc8bb691`). It tracks X and Z frame
dependencies and classical measurement feedback, including nested scopes, and
reports the final live maximum. Debug wires contribute to the touched-qubit
count without contributing to reaction depth.

Validation uses 64 locked reference cases and 24 original C++ assertions. The
remaining controlled-SWAP source test (K-0211) exercises KMX–KMB–KMX transport
with two ordered CSWAP records. The physical stream contains two CSWAP gates.
Its reaction-depth estimate is one. Appending the round-tripped circuit
preserves the complete complex output for both initial classical guard values.
