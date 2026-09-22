# CUDA-Q export contract

FlagQuantum will export immutable FlagQuantum IR to a dynamically constructed
`cudaq.Kernel` through an optional adapter owned by
`flagquantum.ecosystem.cudaq`. CUDA-Q objects stop at that package boundary.
Core, compiler, runtime, simulation, and distributed worker modules must not
import CUDA-Q or expose CUDA-Q types.

The first change establishes the boundary before implementation. It does not
add a public adapter, execute a CUDA-Q kernel, select a CUDA-Q target, or claim
GPU or QPU support. The machine-readable contract fixes a small static-unitary
gate subset, wire mapping, operation order, concrete-real parameter policy, and
fail-closed handling of unsupported operations.

CUDA-Q kernels are more expressive than circuit IR: they can accept arguments,
compose kernels, perform classical computation and control flow, and include
measurements. Arbitrary kernel import would therefore either lose semantics or
require FlagQuantum IR features that do not exist. Version 1 is intentionally
one-way: FlagQuantum IR can become a non-parameterized CUDA-Q builder kernel,
while CUDA-Q-to-FlagQuantum conversion is rejected.

## Dependency decision

The adapter uses the `cudaq` distribution through its own optional extra and
targets the current 0.15.1 and 0.16.0.post1 release lines on Python 3.11 or
newer. CUDA-Q supports Linux on x86-64 and ARM64 and macOS on Apple silicon;
macOS execution is CPU-only. The dependency is deliberately excluded from the
portable `interop-all` aggregate because it installs a platform-specific
compiler and simulator toolchain.

No SDK lane is claimed by this contract-only change. The implementation PR must
install and exercise both declared versions before moving them into the tested
dependency matrix or calling them certified. Removing the adapter and its extra
must leave FlagQuantum IR and Stable Core APIs unchanged.

## Semantic boundary

The exporter will allocate one CUDA-Q qubit per FlagQuantum wire and preserve
operation order. CUDA-Q numbers wire zero as the least-significant statevector
index bit, while FlagQuantum's current statevector convention places wire zero
as the most-significant axis. Conformance must reorder the CUDA-Q statevector
explicitly; equality by raw array position is invalid.

The initial gate map is `X`, `Y`, `Z`, `H`, `S`, `T`, `RX`, `RY`, `RZ`, `CX`,
`CZ`, and `SWAP`. Parameters must already be bound finite real scalars. Identity,
adjoint-only gates, general unitaries, controlled rotations, multi-controlled
gates, channels, measurements, kernel arguments, control flow, composition,
state initialization, and remote execution remain outside version 1.

## Implementation acceptance

The implementation PR must:

1. export the supported FlagQuantum IR subset through `cudaq.make_kernel()`;
2. reject every unsupported opcode and semantic feature with stable diagnostics;
3. reject reverse conversion instead of parsing private Quake or MLIR details;
4. prove fixed-seed statevector equivalence after explicit bit-order conversion;
5. test CUDA-Q 0.15.1 and 0.16.0.post1 in isolated optional lanes;
6. prove importing `flagquantum` does not import CUDA-Q; and
7. register the adapter publicly only after integration and conformance pass.
