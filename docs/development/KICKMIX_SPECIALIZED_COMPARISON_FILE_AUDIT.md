# Specialized comparator file audit

Kickmix reference commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob |
| --- | --- | --- |
| K-0067 | `gen_cmp_low_space.cc` | `7de1b7a8bd5ba36226385d7c319a5260635ebd8e` |
| K-0068 | `gen_cmp_low_space.h` | `02e724e0f1fec912f515faa75644861a93ae703d` |
| K-0069 | `gen_cmp_low_space.test.cc` | `deaa864bbdf49d368962593343c8e171bf220ff8` |
| K-0070 | `gen_cmp_qq.cc` | `7c618915a2b61ea26472652cf88bc79fe17c4096` |
| K-0071 | `gen_cmp_qq.h` | `b53f8e611faf1f192c5c54ee60649c66c0bdea43` |
| K-0072 | `gen_cmp_qq.test.cc` | `88e5b12ae7dd5e4ab90e376a32373678fe3d32c0` |

The comparison adapters in `compiler/registers/comparison/physical_comparison.py`
expose the dedicated quantum and three-clean-wire schedules directly. They
use big-endian words, preserve the quantum enable and equality operands, and
support both target XOR and a scalar sign. The three-wire schedule pads a
short classical right word with zeros on the high side and supports the
source's inverted output. The quantum schedule retains the source's
reversible carry path even when the higher-level dispatcher could instead
choose a full ladder. Its no-workspace narrow branches can borrow a dirty
wire, matching the source's fixed-ID choice and returning it intact.
Classical port IDs are preserved and work slots start after the highest
bound classical input.

The masked-prefix adapter stays with the phase generator in
`compiler/registers/phase/masked_prefix_phase.py`. It applies the source
prefix phases up to the common target/mask length. Clean work precedes
dirty work in the lease. The physical path emits initial resets and measured
cleanup, while the existing append path retains the caller's pre-cleared
contract.

The fixture contains 372 dedicated quantum and 256 three-clean-wire streams
from unchanged C++ and both original masked-prefix KMX/ASCII pairs. Stream
fingerprints exclude only register declarations and the native transport
header. The two diagram cases use entire registers and compare complete KMX
before testing the ASCII. All seven unchanged original tests ran, covering
fuzz, original diagrams, and the source quantum comparator resource bound.
Native exact branch tests cover target XOR and signs, a short right word with
inverted output and a quantum enable, and restoring a dirty wire in a
coherent state. They check unnormalized amplitudes for every reset and HMR
outcome and returned clean work.

| Masked-prefix case | Leased clean | Leased dirty | Emitted CCX | Emitted CCZ | R | HMR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dirty workspace | 0 | 7 | 26 | 2 | 0 | 0 |
| Partial dirty workspace | 2 | 5 | 20 | 2 | 2 | 2 |

The source's separate CCZ-state-weight check is a model. The table counts
actual records and does not convert either gate count into a T estimate.

Regenerate the fixture with
`python tests/unit/fixtures/generate_kickmix_specialized_comparison_file.py --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_specialized_comparison_file.json`.
