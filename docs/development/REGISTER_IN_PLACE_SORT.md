# Reversible in-place sort with permutation output

The register emitter sorts equal-width, big-endian unsigned words in
place and writes one zero-based rank for each original word. The rank at
position i gives the new position of the original word at i. For example,
[6, 1, 3] becomes [1, 3, 6] with ranks [2, 0, 1]. The contract is for
distinct words. Forward takes clean rank words. The adjoint takes the
ascending list and a matching rank permutation, restores the original
list, and clears the ranks. Both directions clear all borrowed quantum
work. Classical slots can retain their most recent measurements.

## Construction and binding

1. Copy the original list into a temporary key array. For lengths that
   are not powers of two, pad to the next power of two with a leading
   sentinel bit set on the padding entries. Data keys have leading bit
   zero, so the maximum data word remains available.
2. Sort the keys through the original BitonicSort network, XOR their
   first L payload words into a separate table, undo the sort, and
   remove the key preparation. The table remains sorted while the
   temporary keys and comparison history are now clean.
3. Use the shared rank XOR to write the original-to-sorted permutation.
   Then XOR the reconstructed original words into the input list to
   clear it, and transfer the sorted table into that cleared list using
   two CX per bit.
4. In the adjoint, reverse those stages. After transferring the sorted
   list into the table, the ranks reconstruct the original input.
   XORing the corresponding ranks clears them, and recomputing the
   sorted copy by XOR clears the table.

For L > 1, the rank width is ceil(log2(L)). Work consists of the table,
padded keys, BitonicSort history, and one shared pool sized for the
larger of the network and indexing stages. The singleton needs no rank
bits, work or gates.

## Reference and resources

[The cited paper](https://arxiv.org/pdf/2406.19378v1#page=34) uses
a clean sorted copy and replaces the original list with permutation
ranks through Lemmas 4.12 and 4.13. The locked
[Qualtran SortInPlace](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/sort_in_place.py)
provides the top-level signature and a model with 3L² each of
Comparator, Comparator adjoint and word XOR. It supplies no executable
decomposition. Tests assemble a reference from the unchanged
BitonicSort source and its adjoint plus the separately verified
permutation primitives. They compare operands, dependency order and
measurement feedback in both directions, including the padded network.
The abstract whole-sort graph and native gate counts are separate.

| List length | Word width | T + T† emitted | CX emitted | Measurements emitted | Clean work qubits | Classical slots |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1 | 38 | 106 | 6 | 7 | 1 |
| 3 | 2 | 684 | 1,608 | 108 | 32 | 5 |
| 4 | 3 | 1,004 | 2,280 | 188 | 38 | 5 |
| 5 | 3 | 3,708 | 8,611 | 591 | 82 | 7 |
| 8 | 4 | 6,080 | 13,312 | 1,184 | 99 | 7 |

Counts are for one direction, including all cleanup. Tests cover every
distinct list in the enumerated small layouts, with a channel
certificate covering all measurement histories. They also exercise
complex amplitudes with a spectator through the append entry point,
including all 64 raw histories per direction at L=2 and one-bit words.
