# Complete command argument source audit

## Interface and ownership before implementation

Frozen Kickmix `util/arg_parse.cc`, `.h` and `.test.cc` are K-0284 through
K-0286 at `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0.
All five source/test-helper blobs are verified and all 11 unchanged original
tests pass in an isolated temporary directory.

Ecosystem owns private command scanning, typed flag conversion, enum reference
selection and edge-owned file acquisition. It consumes plain native strings,
scalars and streams. Core and Simulation own no CLI state. Compiler's existing
binary output boundary retains complete-result writes. The existing KMX command
entry points use argparse for command orchestration and help. Numeric callbacks
can reuse the source integer grammar through its private authoritative owner.
Complete command entry source parents remain separate from this helper group.

Native scanning takes argv including the program name and returns the first
matching flag. It preserves empty/missing distinction, equals syntax, -- and
the source's differing negative-value rules for discovery and unknown checks.
Integer parsing retains sign-only zero and underscore admission after a nonzero
accumulator. Float32 and float64 use the host C runtime strtof/strtod directly,
matching the source runtime instead of double-rounding through Python float.
ctypes is a standard-library dependency. C objects remain private to this
boundary and native functions return only float values. Calls use owned buffers
and retain the C runtime's parse endpoint. No process locale is changed.

Enum lookup returns the original mapped object. A missing default key maps
source out_of_range to KeyError. File acquisition is a context manager that
closes a newly opened handle and preserves caller-owned defaults. Source fopen
binary modes map to Python binary modes. POSIX extended mode admission delegates to the same host C runtime.
Windows text/newline behavior remains a platform-specific follow-up. No quantum primitives,
measurement branches, work bits or quantum resource model belong to parsing.

## Definition disposition and evidence

| Source definition | Native owner and observed behavior |
| --- | --- |
| find_argument and require_find_argument | find_argument/require_argument retain first match, duplicates, terminator, empty values, prefix matching and C-string truncation |
| check_for_unknown_arguments | check_unknown_arguments retains known/deprecated loop order, mode skip, negative-token distinction and sorted diagnostics |
| parse_int64 and find_int64_argument | _command_numbers plus int64_argument retain uint64 intermediate arithmetic, signed bounds, sign-only zero and underscore grammar |
| find_float_argument and find_double_argument | float_argument selects direct strtof/strtod, exact scalar-overload defaults/bounds and signed zero; failure diagnostics match |
| find_bool_argument | bool_argument distinguishes absent, bare and nonempty flags |
| find_enum_argument template | enum_argument retains mapped-object identity and missing-default KeyError corresponding to original out_of_range |
| find_open_file_argument | open_file_argument preserves defaults and standard host-POSIX byte modes with explicit scoped ownership; extended POSIX modes use host fopen |
| Existing KMX commands | sample, spin and sampled-counts use the shared integer callback through their actual main entry points |

Five frozen blobs are checked before compiling the unchanged source. All eleven
unchanged original tests pass in an isolated temporary working directory.
The 2,498-record corpus independently exercises source values, exact diagnostics,
floating-point bits, enum results, file contents/default identity and refusals.
It regenerates byte-for-byte, with hex-encoded observations and a quoted wire
format preserving tabs, newlines and NUL bytes. It includes all standard file
mode spellings exercised here and exclusive-open success/failure. Native tests
also prove default-stream identity, handle closure on success and consumer
failure, enum object identity and actual CLI callback replacement.

Concentrated regression passes 353 tests, and 55 local register-execution,
padding and phase-overlap integration tests pass. Strict mypy passes all seven
changed Python files. Black, Ruff, Ecosystem scope, architecture, internal IR,
protected API baseline and whitespace checks pass. The two private scalar
owners have 99% combined statement/branch coverage. The unreachable defensive
C-result type check is unexecuted. Dependency policy reports 720 existing
whole-repository findings and zero findings in touched paths.

## Source defects and remaining boundaries

The original uint64 overflow predicate accepts `18446744073709551616` as zero
and the next few unsigned wrap aliases. Native preserves the exact predicate,
including rejection of `18446744073709551620`. These are original observations,
not an amended reference. Ordinary signed-bound failures remain distinct.

The float32 witness `1.0000000596046447753906250000000001` rounds directly to
`0x1.000002p+0`, while conversion through Python double then float32 gives one.
The native direct-C route matches the original. NaN explicit values fail,
whereas a NaN default can return under the original comparisons.

On the measured macOS host, original fopen accepts the extended `re` mode.
Native now delegates nonstandard POSIX modes to host fopen, obtains the actual
file access flags, duplicates the untouched descriptor and closes the C stream.
The owned Python binary stream retains initial position, append flags, file
creation/truncation and exclusive-open effects. Its descriptor is non-inheritable.
The default stream is never transferred or closed. C runtime refusal maps to
the existing source diagnostic. Failure to duplicate closes the C stream;
failure to create the Python stream closes the duplicate. The source handle
never escapes Ecosystem, and no native C reads or writes precede transfer.

K-0284's source-success POSIX mode gap is closed. The approved faithful-migration
rule retains Windows text translation and unmeasured runtime portability as
platform follow-ups, not as claims of tested support. Non-POSIX extended modes
still fail explicitly. The existing standard binary mode route is unchanged.
K-0285 and K-0286 have complete declaration and original-test disposition for
the verified native routes. Missing default enum keys follow actual source
out_of_range behavior despite the source header's invalid_argument prose.

Argparse continues to own existing command orchestration/help. Full main_*.cc
scope is separately inventoried. Scoped binary streams adapt original caller
fclose ownership. These helpers are classical edge operations with zero emitted
quantum primitives, measurements or work qubits. There is no declared source
quantum resource model. Accepted full-complex-amplitude, measurement-branch and
clean-work register evidence is reused by the local integration suite.

## Reproduction

```bash
python tests/unit/fixtures/generate_command_arguments.py \
  --source-root /path/to/frozen/kickmix/src \
  --output /tmp/command_arguments.jsonl
cmp tests/unit/data/command_arguments.jsonl /tmp/command_arguments.jsonl
python -m pytest tests/unit/test_command_arguments.py \
  tests/unit/test_kmx_sample_command.py tests/unit/test_kmx_spin_command.py \
  tests/unit/test_kmx_sampled_counts.py -q
python -m flagquantum.ecosystem.kmx_sample --in input.kmx --out output.txt --shots 1__2
```

The last command takes twelve samples using the original signed-integer grammar.
Raw source-run, regression, integration and environment receipts are retained in
`ftqc-reference-env/g2-command-arguments-*20261122.*` beside the main ledgers.


## POSIX host-mode closure (2026-11-28)

The complete current corpus has 2,756 original observations, including extended
read/update/write/append/exclusive mode spellings with `e` in several positions,
invalid modes, existing and absent files, missing/default arguments, empty paths,
file effects and ownership. The five pinned source/test blobs are verified
before compiling the unchanged implementation. Fresh generation reproduces the
corpus exactly; no source mode parser is patched. Both normal and consumer-error
exits close acquired streams, transfer failure closes the duplicate, and the
original caller-owned default remains open. Existing scalar, enum, diagnostics
and actual KMX command callback evidence is reused for unchanged definitions.
This edge-owned implementation adds no Core, compiler or quantum instructions.

Current verification passes 239 concentrated command-helper and actual KMX
command checks plus 82 local register integration checks, without skips. Strict
typing covers all three changed Python files. Black, Ruff, Ecosystem scope,
architecture, IR/API, required checks, Cirq interoperability, English, links and
whitespace pass. The expanded source corpus regenerates byte-for-byte. An
initial regression retained the old fixture cardinality assertion (2,498); all
observations already matched, and the assertion now records the actual 2,756.
Zero quantum primitives, measurements or work qubits are emitted; no independent
quantum model or new performance claim applies to this classical edge helper.
