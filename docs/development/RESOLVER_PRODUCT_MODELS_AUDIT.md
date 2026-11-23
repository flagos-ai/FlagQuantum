# Complete arithmetic product constructor declarations

## Source and ownership

The complete arithmetic/multiplication.py and original tests remain unchanged
at Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. All eight classes,
examples, original callbacks, tensors, cost graphs and notebook assets have an
existing disposition in INTEGER_MULTIPLICATION_FILE_AUDIT.md. The new format
owner legacy_product_model_factories.py supplies explicit constructor mappings
for PlusEqualProduct, Square, SumOfSquares, Product, ScaleIntByReal,
MultiplyTwoReals, SquareRealNumber and InvertRealNumber. It owns argument and
ordered-signature checks only.

All eight unchanged source roots lack a primitive decomposition. Their named
Program entries therefore remain bodyless declarations, and physical lowering
refuses them. Existing native paper-based integer/fixed-point execution owners,
foreign adapters and ordinary circuit.append(op) workflows retain their separate
acceptance. This constructor layer introduces no execution or numerical kernel.
Original classical/tensor behavior and independent analytical graphs continue
to use their existing owners and evidence.

## Ports, directions and symbolic limits

All integer and fixed-point storage is MSB first. PlusEqualProduct has three
THRU words and preserves its forward/adjoint flag. Square retains the RIGHT
result or LEFT erasure port. SumOfSquares retains its k-element array and
2n plus bit_length(k minus 1) result width. Product retains twice the maximum
operand width. ScaleIntByReal retains its mixed integer/fractional formats.
SquareRealNumber retains the original additional b port, independently of the
native paper execution interface. InvertRealNumber retains its source THRU
result and mixed fractional width.

All eight concrete examples round-trip through the unchanged original wire.
Supported symbolic widths retain their exact declarations. Original
SumOfSquares with symbolic k raises at bit_length. Original Product with
independent symbolic widths raises at Python max. Even equal positive source
symbols lose their assumptions in original wire text, and the restored original
Product signature then raises at max. This counterexample is preserved in both
the original restored object and native constructor, without changing the
reference. Existing fixed-point validators and Proposal 120 declaration rules
remain authoritative.

## Phase, work and resources

No primitive or workspace is emitted by a declaration. Missing root expansion
has an explicit physical-lowering refusal. Existing accepted paper-defined
execution evidence retains full complex amplitudes, clean work and small raw
measurement histories at its recorded scope. Independent native metadata and
resource regressions cover all eight original direct call graphs and leaf
counts. Source estimates remain independent of measured execution inventories,
including negative small-width model estimates and exact-swap substitutions.
No zero-gate physical implementation is inferred from a bodyless declaration.

These eight constructor mappings close their original declaration scope and
raise local catalogue coverage from 77/303 to 85/303 after publication. The
whole arithmetic file remains open for missing upstream primitive references,
foreign protocol/rebinding and the native fixed-point square/reciprocal contract
differences already registered in the whole-file audit. Q-0876 remains open.
Receipts use ftqc-reference-env/g2-product-model-constructor-*.

## Local batch acceptance

The concentrated native regression passes 224 cases, including all 21 focused
constructor checks and existing whole-file metadata, foreign, phase, cleanup and
resource evidence. Local integration passes 145 cases. The unchanged original
source test file passes 24 cases, with 24 original no-decomposition/count/type
skips and one notebook deselected. The earlier complete notebook intake remains
reusable. Seven frozen source blobs and real Kickmix 0.1 Simulator are verified.
Both Python files pass strict typing, Black and Ruff. Compiler scope, language,
architecture, IR and protected API checks pass. The dependency baseline remains
identical at 720 findings. Local acceptance precedes publication verification.
