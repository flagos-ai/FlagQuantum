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

## Binary file helpers

The locked `binary_file_tools.cc` and `.h` (K-0287/K-0288, blobs
`0d1f159d224db05085e901b0663f4d647c3a6d42` and
`291c48f3985afb467ec30ac29993bc1301c8210c`) use little-endian scalar and
block writes despite the `_be` suffix. The codec's explicit little-endian
formats handle the byte swaps through Python's `struct` and share the
scalar formats across packet and register fields. The byte reader checks
the full requested count before advancing. The converter checks the number
of output bytes and reports write and flush failures.

An unchanged C++ probe checked scalar writes, byte swaps, zero-length I/O,
short reads, write refusal and blocks of 0, 1, 127, 128, 129 and 257 words.
Native codec tests replay those lengths against complete KMB digests from
the locked extension, including truncation inside each nonempty block.
The C++ probe runs on a little-endian host. The native explicit format is
independent of host byte order.

The source temporary-file helper opens two files and loses the first
handle, including when the second open fails. A call-count probe confirmed
the leak. FlagQuantum converts the in-memory payload directly and opens
only caller-selected paths. The broader Python circuit binding and diagram
commands remain tracked in their own source-file rows.
