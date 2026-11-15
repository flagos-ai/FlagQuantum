# Chemistry black-box source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen chemistry files
are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0335 | `black_boxes.py` | `f9670a2ebaf2fe0973323cf7b532c88142970902` | Concrete resource formulas and controlled-number phase verified. General foreign-object protocol and source wire-symbol callbacks remain open. |
| Q-0336 | `black_boxes_test.py` | `23cfe1e1bae3f437837e031348a0d09bfe201bf0` | Closed for all four original OpenFermion comparisons and the corresponding native resource queries. |

## QROAM estimates

The Algorithms resource owner keeps the unrounded clean and dirty block
scores and the one- and two-axis integer Toffoli estimates. It reuses the
Compiler's clean forward block-exponent selection. The two directions
select block size independently, and explicit block size one keeps the
source's rows-minus-one rule. The two-axis placeholder uses both axes
in the cost but only the first axis in the signature. Its selection
wire width is `(first - 1).bit_length()`; a zero-width selection is
omitted. The target is a through word in both directions.

The four original OpenFermion comparisons ran unchanged in an isolated
OpenFermion 1.6.1 environment. The native tests check the same four
data-size and target-width pairs in both directions against the locked
Qualtran formulas. They also check explicit blocks, concrete port order,
single-row selection omission, two-axis cost and the dirty score.
The source QROAM types are resource placeholders. Their estimates supply
no physical gate stream or numerical lookup evidence. Executable lookup
stays with the existing Compiler table and erasure components. The
declared dimensions are concrete; no symbolic extension is inferred.

## Controlled number phase

The Algorithms binding keeps the source control polarity, control-array
shape and through system register. It acts on the most significant system
wire and leaves the remaining system wires untouched. The caller owns
the predicate work wires and classical result slots. The Compiler
MultiControlZ emitter supplies gates and feedback. DF and THC use the
same resource declaration owner.

| Controls | Emitted predicate plus phase | Original wrapper estimate |
| --- | --- | --- |
| One | CZ with any negative-control X pair, zero T or measurement | Zero Toffoli |
| Two | Four T/T-dagger, one measurement, one work wire | One Toffoli |
| Three | Eight T/T-dagger, two measurements, two work wires | Two Toffoli |

The estimates above retain the source wrapper's explicit Toffoli-only
call graph. All matched primitive streams, including measurement
feedback, are checked independently. Complete complex states spanning
all control and system values agree with the expected sign pattern in
every applicable history. Work returns to zero. Native binding requires
a nonempty system and at least one control. The source wire-symbol
presentation and arbitrary external Bloq interoperability remain open.
