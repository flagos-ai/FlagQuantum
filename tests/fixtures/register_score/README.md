# Register score reference asset

`example.json` is the unmodified `qualtran/drawing/musical_score_example.json`
from Qualtran v0.7.0, commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
Its Git blob is `b5ed08cad72ae69d275fc89b90337b7dc4daf3a8` and its size is
4,681 bytes. Source copyright belongs to Google LLC, licensed under Apache-2.0.
Source https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score_example.json

License https://www.apache.org/licenses/LICENSE-2.0

This original browser asset has horizontal lines without flavors and string
connector labels. Current Python-generated MultiAnd records are a separate test
input. Keep the bytes intact so hash verification can distinguish the two.

The owning tests are `tests/unit/test_drawer_score_asset.py` and the original-asset
parameters in `tests/integration/test_drawer_score_loading.py` and
`tests/integration/test_drawer_score_domains.py`. They exercise offline embedding,
HTTP import and the original HTML entry point against the frozen source script.
