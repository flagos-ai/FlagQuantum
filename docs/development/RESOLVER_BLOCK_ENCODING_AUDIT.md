# Typed SELECT and LCU constructor binding

Reference: unchanged Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`, Apache-2.0, Google LLC.
The source/test blobs are `679ac7964baf62288713078a5ec4dbce674cbece`
and `0632799611bbd5ec2b7ca724533d4bbd1bbb7169`.

## Ownership and executable path

Compiler preparation owns the typed SELECT-only and prepare/SELECT/unprepare
graphs in `block_encoding_component.py`. Existing Algorithms signature helpers
delegate to that owner. SELECT-only executes one SELECT and retains PREPARE as
signal metadata; the LCU zero-signal state remains the existing identity PREPARE.
Formats resolves native oracle roles and exact forward/inverse pairs. Ecosystem
shares scoped packed/Hubbard child owners with the existing packed constructors.
No SDK objects enter Compiler, and no numerical execution or backend selection
is added there. Users still append the bound operation to their circuit.

Selection, target and THRU preparation junk are caller-owned MSB-first words.
RIGHT preparation junk is internal to the block: it survives SELECT and is
consumed by exact unpreparation. Each child owns its private clean workspace.
The existing LEFT-junk input-extraction failure remains a separate source gap.

Positive Hubbard control calls its specialized SELECT with port `control`.
Zero control calls the original `ControlledViaAnd` fallback with port `ctrl`;
the X flips stay inside that child. The root never substitutes an opaque
controlled matrix or moves the flips across preparation. Existing shared
control factories own the fallback; the Ecosystem adapter supplies native
positive forward/inverse children. Generic adjoints reverse typed calls using
explicit exact pairs, including the control wrapper.

Packed wrapper properties and public signatures retain their distinct nominal
types. In particular, a one-bit PREPARE signature can use QBit while its
AutoPartition child and selection property use QAny(1). The shared packing
owner now derives the latter from the original group widths. This changes no
public signature, physical gates or workspace requirement.

## Evidence and replacement

`tests/unit/test_legacy_block_encoding_factories.py` resolves all four original
plain/packed examples and compares root metadata and named call operands.
Original controlled Hubbard roots are compared recursively down to already
audited atomic children. Independent native PREPARE and SELECT owners replace
the original oracle implementations without changing the block consumer.
The small non-Hermitian fixtures cover absent/zero/positive control, THRU and
RIGHT junk, both directions, complete complex operators, coherent roundtrips,
every combined raw measurement history and zero work at return. A packed
one-bit case checks the nominal-type distinction through recursive source calls
and its full complex operator. SELECT-only proves that PREPARE is not executed.
Source imports are blocked after serialization and compilation/binding/append
still succeed through native owners.

Batch receipts live in `ftqc-reference-env/g2-lcu-block-constructor-*` beside
the worktree. The intake verifies both frozen blobs. Resource receipts verify
Qualtran 0.7.0 and an actual Kickmix 0.1 Simulator, and list emitted primitive
names and live private leases independently from QEC and logical-qubit models.
Supplied pool size is not live peak; lease release alone is not amplitude-zero
evidence. The full operators and raw-history tests supply the latter.

## Retained gaps

- The former ordinary Hubbard SELECT scalar-body gap is closed through the
  existing private carrier. All four original SELECT/LCU forward factories and
  both Hubbard specialized control paths now bind named native primitive bodies.
  See [the active dependency closure](HUBBARD_PREPARATION_CONSTRUCTOR_AUDIT.md).
  Proposal 119's new Core opcode remains pending and is not used.
- Original LCU inverse serialization and deeper plain SELECT inverse
  serialization fail on `SpecializeOnCtrlBit.ONE`. The source is not patched;
  native exact-pair tests do not imply that this original wire path succeeds.
- Arbitrary oracle role contexts and specialized control adders must be supplied
  explicitly. Roles, inverses and semantics are never inferred from counts.
- LEFT junk, symbolic physical layouts, original missing packed-control ports
  and large exhaustive operators remain separately tracked. This batch does
  not close Q-0283 or the generic constructor catalogue parent.

Run the focused scenario from the repository root in the locked reference
environment, with torch imported before pytest:

```bash
python -m pytest tests/unit/test_legacy_block_encoding_factories.py -q
```
