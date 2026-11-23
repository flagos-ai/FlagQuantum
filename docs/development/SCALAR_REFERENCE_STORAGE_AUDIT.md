# Frozen scalar reference storage

Five complete oracle payloads now use XZ compression to meet the existing tracked
file-size limit. The original 24,165,943 payload bytes decompress exactly from
2,097,040 tracked bytes. All five compressed files remain below the existing
1,800,000-byte individual limit. Standard-library lzma owns compression and
streaming reads. The repository hygiene limits retain their existing values.

| Original payload | Original bytes | Compressed bytes |
| --- | --- | --- |
| fixed_width_integer_reference.jsonl | 7,870,040 | 854,688 |
| modular_integer.jsonl | 2,082,925 | 104,348 |
| packed_chacha.jsonl | 8,536,563 | 1,064,688 |
| packed_word.json | 2,710,938 | 32,912 |
| population_counter.jsonl | 2,965,477 | 40,404 |

The storage integrity test binds the complete decompressed size and SHA-256 of
each original payload, computed from the pre-change Git objects. It streams the
hash in 64 KiB chunks. Existing original-source observation tests still execute
every retained record. Integer and modular input readers accept both plain and
compressed paths, preserving generator input behavior. JSONL consumers stream
records. The packed-word consumer retains its existing complete JSON loading
behavior. Regeneration tools continue to emit their original plain payloads.
To update storage after an explicitly reviewed oracle regeneration, compress the
plain payload with lzma.compress and verify the decompressed bytes before replacing
the tracked archive. An intentional changed observation requires review of the
integrity size/hash as well as the original source receipt.

This changes test corpus storage only. Source implementations, primitive streams,
complex amplitudes, work ownership, resource models and frozen oracle payloads
retain their existing evidence. The byte-preservation receipt is stored beside
the worktree at ftqc-reference-env/g2-scalar-storage-payloads-20261123.json.
