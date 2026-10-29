# Fixed Classiq source corpus

These three original Qmod files and the MIT license are restored byte for byte
from the assets retained when PR #949 was closed. The manifest pins upstream
Classiq commit `2bfd835e73b6882fcc515a725098b04766c238e7`, original paths, Git
blob identities and SHA-256 digests. The retained FlagQuantum source revision
was `0dc6875ede9c0313d4350e93a5e06b429bb7d4e9`. The superseded implementation,
public API proposal and per-file circuit fakes from that PR are not restored.

Run from the repository root:

```bash
python tests/fixtures/classiq_qmod/verify_sources.py
```

Expected: three verified source names, the original MIT license, and seven
digest checks passed. This checks source integrity only. It does not parse,
lower, execute or verify any original Qmod program. No SDK, network, credentials
or Classiq service is needed.

See the [source admission record](../../../docs/roadmap/CS07_SOURCE_ADMISSION.md)
and [shared boundary contract](../../../docs/api-changes/FQ-SYNTH-CS07-20261029.md).
Reduced-width, indexed and negative diagnostic scenarios must live in tests and
be labelled synthetic; never edit these originals or count those scenarios as
additional original programs.
