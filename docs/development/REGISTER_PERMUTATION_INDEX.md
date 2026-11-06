# Reversible permutation indexing

These two register emitters implement the indexing and reconstruction
steps in [Lemma 4.12, Eq. 122](https://arxiv.org/pdf/2406.19378v1#page=33).
The caller binds a table of distinct words, a permutation of the table,
and zero-based rank words. All registers are big endian. For a table of
length L, the rank width is ceil(log2(L)). The singleton uses an empty
rank word. Table words may be in superposition.

- `permutation_rank_xor_ir` preserves the table and source words.
  It compares each source word against each nonzero rank's entry,
  XORs the matching rank into its destination, and clears the match flag.
  Rank zero contributes an empty mask.
- `permutation_words_xor_ir` preserves the table and ranks. It compares
  each rank against concrete indices, XORs the selected quantum entry into
  its destination, and clears the flag. Unused rank patterns select zero.

Together the operations map an original word list and clean rank words to
a clean word list and the corresponding ranks, preserving the distinct
table. Reversing their order reconstructs the original list and clears
the ranks. Both entry points also accept occupied XOR targets. The
single borrowed pool holds the match flag, equality work and one
reused controlled-X carry. Scratch qubits return to zero after every
lookup. Classical slots hold the last measurement outcomes.

The [locked SortInPlace source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/sort_in_place.py)
provides the concrete rank signature and an abstract whole-sort graph
with 3L² each of Comparator, Comparator adjoint and word XOR. It has no
gate decomposition. The executable reference test builds the two
indexing streams from the unchanged Equals, EqualsAConstant, And and
And adjoint leaves and compares their operand order and feed-forward.
The following counts apply to the emitted indexing stages.

| Table length | Word width | Rank width | Work qubits | Classical slots | Rank T + T† / CX / measurements | Reconstruct T + T† / CX / measurements |
| ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| 1 | 2 | 0 | 0 | 0 | 0 / 0 / 0 | 0 / 2 / 0 |
| 2 | 1 | 1 | 2 | 1 | 0 / 14 / 0 | 16 / 36 / 4 |
| 3 | 2 | 2 | 2 | 1 | 48 / 138 / 12 | 144 / 252 / 36 |
| 4 | 3 | 2 | 3 | 2 | 192 / 472 / 48 | 320 / 560 / 80 |
| 8 | 4 | 3 | 4 | 3 | 1,344 / 3,120 / 336 | 2,048 / 3,456 / 512 |

Coverage includes every permutation of every small distinct sorted
table in the test layouts, occupied destinations, unused rank patterns,
and complete complex amplitudes with a spectator in all 16 raw
measurement histories of the two-entry circuit appended to a circuit.
