# Locally revised Classiq Qmod acceptance corpus

The base is Classiq commit `2bfd835e73b6882fcc515a725098b04766c238e7`.
Its 216 pinned inputs have 210 exact CPU execution smoke passes and six
semantic refusals. The original bytes and result count are retained.
This user-approved local overlay replaces six input vectors or success
amplitudes in a separate acceptance corpus. The other 210 inputs use
their pinned bytes. The overlay has its own hash and is a FlagQuantum
acceptance revision of the corpus.

Use the unchanged corpus checkout and its tracker as the first argument.
The materializer verifies all upstream Git blobs, applies the patch in a
temporary directory, and verifies every resulting source before publishing
the separate corpus directory.

```sh
overlay=tests/fixtures/classiq_qmod/accepted_revision
PYTHONPATH=. python tools/materialize_qmod_revision.py \
  /path/to/pinned-corpus "$overlay/changes.patch.gz" "$overlay/manifest.json" \
  /path/to/local-acceptance-corpus
PYTHONPATH=. python tools/audit_qmod_corpus.py \
  /path/to/local-acceptance-corpus /tmp/qmod-acceptance.json --execute \
  --bindings "$overlay/bindings.json" --basis-mps-paths "$overlay/basis-mps-paths.json"
```

The Classiq sources and proposed edits to them use the neighboring
[`LICENSE.txt`](../LICENSE.txt). The manifest records both original
and revised file digests. The synthesizer applies its normal checks to the
bytes it receives.
