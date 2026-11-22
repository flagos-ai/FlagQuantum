# Complete shared test support source audit

## Ownership and interface before implementation

Frozen Kickmix test.test.cc and .h are K-0326/K-0327. Verification owns
filesystem lookup, rewind/read/close, scoped temporary assets and per-test random
stream fixtures. Simulation owns standard seed_seq initialization of the existing
mt19937_64 numerical engine. Existing scalar-seeded callers retain their streams.
No production public API is introduced.

The native entropy input is an explicit iterator of uint32 draws. External
seeding consumes eight words immediately. Each independent stream consumes four
parent uint64 words immediately, truncated to uint32 by seed_seq. The parent is
lazily initialized once per fixture owner. Tests own the lifetime instead of
sharing mutable process globals. Numerical state uses 312 uint64 words and
624 little-endian uint32 initialization words.

Lookup tries all three original prefixes for existing readable files before
trying any writable prefix. It returns original relative spelling and creates an
empty file on the write path. Binary stream helpers close on both success and
failure. Temporary assets use pathlib with scoped descriptor/path cleanup.

## Original boundary

The source temporary-file destructor unlinks its path but leaves its original
mkstemp descriptor open. The native context closes the descriptor and unlinks
on scope exit. This ownership correction is explicit. K-0326 remains open for
the original descriptor leak. The original source remains unchanged.

External random_device results are nondeterministic. The source harness runs
that route for a shape/success check. Deterministic seed_seq and original
independent stream comparisons use explicit source parent seeds, with their
complete draw order recorded separately. They do not claim identical OS entropy.

These classical test helpers emit zero quantum primitives, measurements or
work bits. No analytical quantum model is declared. Existing circuit.append(op)
complex-state, branch, clean-work and actual primitive evidence is reused.

## Complete definition disposition and evidence

| Original definition | Native owner and verified behavior |
| --- | --- |
| resolve_testdata_file_path | Verification helper preserves all-read-before-write priority, returned spelling, empty creation and original failure text. 64 complete layouts and readable-directory acquisition agree with original fopen. |
| externally_seeded_rng | Eight explicitly owned entropy draws feed standard seed_seq. Original OS route is executed, with no exact entropy pairing claim. |
| INDEPENDENT_TEST_RNG | One test-local lazy parent, four eager uint64 parent draws per child and standard uint32 truncation. Three children per original parent retain full draw order. |
| rewind_read_close | Existing binary stream is rewound, fully read and closed. Native context additionally closes after seek/read failure. |
| RaiiTempNamedFile constructors | Standard tempfile creates an empty or binary-filled scoped asset. pathlib owns read/write operations. |
| read_contents / read_bytes / write_contents | All 256 byte values, complete replacement/truncation and empty files preserve observed bytes. Native filesystem exceptions replace source runtime_error wrappers. |
| destructor and descriptor | Original path removal plus leaked original descriptor is reproduced. Native context deliberately closes and removes its asset, including failed test scopes and paths already removed. |
| Static shared RNG state | Explicit fixture-owned stream replaces process mutable globals. Child order is independent of subsequent child consumption order. |
| Header declarations | All declared routes have an owned native test equivalent. K-0327 is accepted; original defect remains recorded on K-0326. |

The frozen two-file blobs are verified before compiling an include-wrapper of
the unchanged source. No token substitution or entropy interception is used.
Nine standard seed vectors cover empty input, low/high words, uint32 truncation,
eight-word seeds and input longer than the 624-word state. Four original scalar
parents each produce three complete independent children. Each stream compares
630 words, crossing two mt19937_64 state boundaries. Source file lookup and
creation are replayed in isolated working directories. The observation corpus
regenerates byte-for-byte.

Concentrated regression passes 147 tests, including scalar modular sampling,
packed ChaCha, counters and fixed-width integers. All 55 local register integration
tests pass. Strict mypy passes four Python files. Native fixture statement/branch
coverage is 100%; the shared numerical RNG owner has 97% coverage, with the
standard all-zero-state repair branch unobserved. Combined coverage is 98%.
Black, Ruff, team scope, architecture, internal IR and protected API checks pass.
Architecture-specific random_device and filesystem portability remain separate
from this host's observations. No new production dependency is introduced.
