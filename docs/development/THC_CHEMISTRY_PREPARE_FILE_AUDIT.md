# Tensor hypercontraction PREPARE audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/thc`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0415 | `__init__.py` | `ed618a10f28b3d87bb838ddfe308e2d2473a85e9` |
| Q-0416 | `notebook_utils.py` | `a07567e3f8f1d002af00512d9f4a38024edb364f` |
| Q-0417 | `prepare.py` | `f93891e72790d0e86f1b73c6a1c1c49089d2550c` |
| Q-0418 | `prepare_test.py` | `8f3da8352c3cd76aff25291b1f9900380a3a9660` |

The source package's three exports map to the native
`PrepareTHCChemistry`, `THCUniformPreparation`, and `SelectTHC`
owners. These are composed by the chemistry walk. The frozen
notebook utility supplies QROM display labels and Cirq Clifford
generalization for source documentation. Its native graph
presentation remains part of the THC tutorial batch. This keeps
notebook presentation separate from circuit lowering.

## Register and resource contracts

The uniform component has two opaque through orbital words and
three through flags. The signed alias preparation has bounded
orbital words and selection flags, a through contiguous index and
two auxiliary flags, and right-only lookup outputs. The five
output datasets retain their source names, dtypes, order, and
bank shape. Tests compare both forward and adjoint source
signatures and each group at block exponents zero, one and two.

At exponent zero the source explicitly declares five zero-length
bank axes. The native source signature retains those declarations.
The physical layout binds only the five populated output words,
so the existing unblocked forward and coherent adjoint still run.
The locked source peak calculation cannot construct this case
because the decomposition requests `junk_target0_`. The gate-only
query remains available and matches the source.

The uniform-state and signed PREPARE execution use the existing
native triangular and one-body index convention. The source
cost model retains the one-body sentinel rule and fails for ranks
1, 3 and 7. The locked circuit's mutually exclusive tests and
non-executable greater-than leaf prevent treating its complete
gate stream as the successful native preparation. Signed PREPARE
retains source-ordered coefficients for the model and maps whole
rows to contiguous physical addresses. The execution checks cover
exact complex phase from blocked bank exchanges, coherent alias
choices, full output amplitudes across representative feedback
histories, and clean temporary work. Lookup tests exhaust the
applicable smaller measurement histories.

The unchanged source PREPARE tests under Protobuf 5.29.5 returned
13 passes, three skips, one expected failure and one notebook
deselection. Protobuf 5.29.5 permits the original serialization
case. The original 40-cell notebook passed independently after
restoring the pinned bundled FlameGraph script, Git blob
`8c917ecf323e13c1ab3c1107c1963f33dec0123e`. The initial missing
file was a reference-environment gap.

Native regression passed 150 cases across THC, QROAM and related
resource models. Strict typing, format, scope, architecture, IR,
and public API checks passed. Q-0415 and Q-0418 close the package
mapping and original tests. Q-0416 remains for the native
tutorial utility. Q-0417 remains open for source sentinel,
unblocked peak and original decomposition boundaries.


## Whole-source disposition, 2026-11-28

The subsequent [literal preparation review](THC_PREPARATION_SOURCE_AUDIT.md)
now accounts for the complete Q-0417 source, including its previously missing
constructor graphs and exact explicit refusals. The Algorithms integral owner
also preserves original float32 alias rounding and contractions; existing
float64 records remain unchanged. Under the approved faithful-migration rule,
Q-0417 closes with original source defects recorded separately. The earlier
hold above is historical, not a claim that the original circuit now physically
prepares the intended state. Original model-only leaves remain non-executable;
large native Walk context is still separate Q-0422/Q-0658 work.
