# Cirq qubit manager source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The source and test blobs are
`da3e8d2ee92618a6c46c9c3e8ffbfc9d200dad07` and
`c0132d612e4a90972c5986b5cc77f47a67a0743e`.

The optional Ecosystem wrapper implements Cirq's three manager methods and
delegates clean allocation, dirty borrowing and release to an injected base.
It defaults to the source's Simple allocator.
Caller-owned wires are reserved separately. If a base manager offers a
reserved wire, the wrapper temporarily holds that candidate until it can
finish the request, then returns it to the base. The same logic handles
clean and borrowed requests. Cirq controls allocation dimensions and the
base reuse policy. No Cirq Qids enter native IR.

The one-step decomposition protocol reserves its original operands before
running a Cirq gate or operation. The collision fixture forces a matching
greedy pool name and compares the exact two CNOTs emitted on the caller input
and a separate work wire. It compares the full complex state of the two-wire
roundtrip and verifies the work wire returns to zero. The manager itself
emits no gates and contributes no gate-count model. Mapping introduced
external work into native owned lifetimes remains an import responsibility.

## Verification and source differences

- Both original manager fixtures pass against Simple and Greedy bases.
  Delegated frees, external reservations and repeated unknown frees retain
  the source behavior. Allocator collisions are compared against the locked
  implementation for both clean and borrowed requests.
- The locked implementation reads a release iterable twice. With a generator,
  ordinary work is consumed on the first read and never forwarded to the base.
  The native wrapper snapshots the iterable and forwards every internal wire
  exactly once. The mixed-release fixture verifies complete greedy reuse.
- A failed delegated release retains external reservations. When allocation
  fails after holding a skipped candidate, that candidate is returned before
  propagating the error.
