# KMB circuit transport

The KMB v2 codec in `compiler/formats/kmb.py` implements the transport portion
of the locked Kickmix `circuit.cc` (K-0014, blob
`fa9ce4093c1e37c416774b1d87447ad73d64a79d`). The complete opcode tables follow
`op_type.cc` and its header (K-0027/K-0028, blobs
`e5bf735308967028b5d9de9ccd39c912c956453a` and
`65560d5bc9d42691a2549d914a2cfb3792d7cc3f`). Other responsibilities of
`circuit.cc` remain under review.

The numeric fields are little-endian, including the two exact phase-angle words.
The codec preserves physical operand order, registers, tags and all 39 opcode
bytes. Text names and tags use UTF-8. It checks packet order, fixed-size payloads,
indices, opcode flags, truncation and trailing bytes before returning canonical
KMX. Execution admission remains owned by the KMX decoder.

Five locked fixtures verify both byte directions, including all operation bytes.
The unchanged upstream `testdata/big_circuit.kmb` (K-0328, blob
`bd9e8b93ed34b808a7398790b1ceeb6bce462e35`) was also checked. Its 263,555
canonical KMX lines match the locked Simulator and re-encode to the original
2,207,593 bytes. The fixture SHA-256 is
`b9dc2b15a29616ae4b1f1a68412f8754ec2c8896c468fc8d294facd7635a69eb`.
Transport has no execution gate cost.
