# Frozen constructor catalogue registration audit

This is the complete 303-key snapshot of Qualtran v0.7.0
`serialization/resolver_dict.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`b10adb30b493afda81fcff2e330618415df655a2`.
This table inventories Compiler format providers, including their explicit
context callbacks. Ecosystem providers can already resolve keys absent here;
a missing format registration is not evidence of missing functionality. Hubbard
PREPARE, for example, already had an Ecosystem constructor before its shared
Compiler ownership was added.

It records exact identifiers and actual defining modules, rather than guessing
source ownership from a package alias. Factory providers requiring context are
queried with nonexecuting callbacks; no constructor or dependency callback is
executed by this registration inventory.

There are 182 exact Compiler format registrations, no canonical-only aliases,
and 121 keys without an exact Compiler factory. Those
121 include the CompositeBloq graph special case, the CtrlSpec argument
descriptor, abstract interfaces and one original module-valued entry. They
are not all missing executable constructors. Generic connected graph resolution
is separate from shallow named-constructor reconstruction.

**Registration does not certify acceptance.** The old 117/303 complete total
is historical. Subsequent scoped module receipts remain valid but have not yet
been reconciled into a current global complete total. Rows marked pending must
be reconciled against those receipts before either acceptance or missing
implementation is asserted. No parent checklist item closes from this table.

The three Hubbard rows have current scoped constructor and exact scalar evidence
in [the Hubbard audit](HUBBARD_SELECT_STORED_GRAPH_AUDIT.md). The two Ising
constructor rows have current full-file evidence in
[the Ising audit](ISING_TROTTER_FILE_AUDIT.md). Its original one-site ZZ failure,
zero/symbolic dimensions and native validation boundary remain explicit.

The original key `qualtran.bloqs.chemistry.trotter.trotterized_unitary`
resolves to a module, not the TrotterizedUnitary class. Four modular arithmetic
package aliases now reuse their authoritative addition/subtraction factories;
[their wire audit](MODULAR_PACKAGE_EXPORT_AUDIT.md) distinguishes successful
legacy reads from retained source canonical-name failures and documents the
shared ModAdd modulus-view bridge.

## One-to-one inventory

The factory column lists native format owners. `Unregistered` means no factory
provider mapping, not proof that the connected graph or algorithm is absent.

| Frozen identifier | Defining source | Exact registration | Acceptance evidence |
| --- | --- | --- | --- |
| `qualtran._infra.adjoint.Adjoint` | `qualtran/_infra/adjoint.py` | Registered: `legacy_alias_preparation_factories.py, legacy_block_encoding_factories.py, legacy_chebyshev_factories.py, legacy_conjunction_factories.py, legacy_hamming_weight_factories.py, legacy_linear_combination_factories.py, legacy_majorana_factories.py, legacy_packed_oracle_factories.py, legacy_phased_block_factories.py, legacy_predicate_factories.py, legacy_qubitization_qpe_factories.py, legacy_qubitization_walk_factories.py, legacy_relative_swap_factories.py, legacy_rotation_preparation_factories.py, legacy_select_swap_factories.py, legacy_sparse_chemistry_factories.py, legacy_sparse_matrix_factories.py, legacy_sparse_preparation_factories.py, legacy_sparse_rotation_preparation_factories.py, legacy_swap_network_factories.py, legacy_thc_preparation_factories.py, legacy_uniform_preparation_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran._infra.composite_bloq.CompositeBloq` | `qualtran/_infra/composite_bloq.py` | Connected graph special case | Published module receipts pending reconciliation |
| `qualtran._infra.controlled.Controlled` | `qualtran/_infra/controlled.py` | Registered: `legacy_chebyshev_factories.py, legacy_phased_block_factories.py, legacy_qubitization_walk_factories.py, legacy_sparse_matrix_factories.py, legacy_word_factories.py` | Published module receipts pending reconciliation |
| `qualtran._infra.controlled.CtrlSpec` | `qualtran/_infra/controlled.py` | Argument descriptor | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.addition.Add` | `qualtran/bloqs/arithmetic/addition.py` | Registered: `legacy_arithmetic_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.addition.AddK` | `qualtran/bloqs/arithmetic/addition.py` | Registered: `legacy_arithmetic_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.addition.OutOfPlaceAdder` | `qualtran/bloqs/arithmetic/addition.py` | Registered: `legacy_arithmetic_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.bitwise.BitwiseNot` | `qualtran/bloqs/arithmetic/bitwise.py` | Registered: `legacy_word_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.bitwise.Xor` | `qualtran/bloqs/arithmetic/bitwise.py` | Registered: `legacy_word_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.bitwise.XorK` | `qualtran/bloqs/arithmetic/bitwise.py` | Registered: `legacy_word_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.BiQubitsMixer` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.CLinearDepthGreaterThan` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.Equals` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.EqualsAConstant` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.GreaterThan` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.GreaterThanConstant` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.LessThanConstant` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.LessThanEqual` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.LinearDepthGreaterThan` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.LinearDepthHalfGreaterThan` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.LinearDepthHalfGreaterThanEqual` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.LinearDepthHalfLessThan` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.LinearDepthHalfLessThanEqual` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.comparison.SingleQubitCompare` | `qualtran/bloqs/arithmetic/comparison.py` | Registered: `legacy_comparison_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.controlled_add_or_subtract.ControlledAddOrSubtract` | `qualtran/bloqs/arithmetic/controlled_add_or_subtract.py` | Registered: `legacy_arithmetic_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.controlled_addition.CAdd` | `qualtran/bloqs/arithmetic/controlled_addition.py` | Registered: `legacy_arithmetic_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.conversions.contiguous_index.ToContiguousIndex` | `qualtran/bloqs/arithmetic/conversions/contiguous_index.py` | Registered: `legacy_contiguous_index_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement.SignedIntegerToTwosComplement` | `qualtran/bloqs/arithmetic/conversions/ones_complement_to_twos_complement.py` | Registered: `legacy_signed_encoding_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.conversions.sign_extension.SignExtend` | `qualtran/bloqs/arithmetic/conversions/sign_extension.py` | Registered: `legacy_sign_extension_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.conversions.sign_extension.SignTruncate` | `qualtran/bloqs/arithmetic/conversions/sign_extension.py` | Registered: `legacy_sign_extension_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.hamming_weight.HammingWeightCompute` | `qualtran/bloqs/arithmetic/hamming_weight.py` | Registered: `legacy_hamming_weight_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.lists.has_duplicates.HasDuplicates` | `qualtran/bloqs/arithmetic/lists/has_duplicates.py` | Registered: `legacy_sorted_duplicates_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.lists.sort_in_place.SortInPlace` | `qualtran/bloqs/arithmetic/lists/sort_in_place.py` | Registered: `legacy_in_place_sort_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.lists.symmetric_difference.SymmetricDifference` | `qualtran/bloqs/arithmetic/lists/symmetric_difference.py` | Registered: `legacy_symmetric_difference_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.multiplication.InvertRealNumber` | `qualtran/bloqs/arithmetic/multiplication.py` | Registered: `legacy_product_model_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.multiplication.MultiplyTwoReals` | `qualtran/bloqs/arithmetic/multiplication.py` | Registered: `legacy_product_model_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.multiplication.PlusEqualProduct` | `qualtran/bloqs/arithmetic/multiplication.py` | Registered: `legacy_product_model_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.multiplication.Product` | `qualtran/bloqs/arithmetic/multiplication.py` | Registered: `legacy_product_model_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.multiplication.ScaleIntByReal` | `qualtran/bloqs/arithmetic/multiplication.py` | Registered: `legacy_product_model_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.multiplication.Square` | `qualtran/bloqs/arithmetic/multiplication.py` | Registered: `legacy_cost_phasing_factories.py, legacy_product_model_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.multiplication.SquareRealNumber` | `qualtran/bloqs/arithmetic/multiplication.py` | Registered: `legacy_product_model_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.multiplication.SumOfSquares` | `qualtran/bloqs/arithmetic/multiplication.py` | Registered: `legacy_product_model_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.negate.Negate` | `qualtran/bloqs/arithmetic/negate.py` | Registered: `legacy_arithmetic_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.permutation.Permutation` | `qualtran/bloqs/arithmetic/permutation.py` | Registered: `legacy_permutation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.permutation.PermutationCycle` | `qualtran/bloqs/arithmetic/permutation.py` | Registered: `legacy_permutation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.sorting.BitonicMerge` | `qualtran/bloqs/arithmetic/sorting.py` | Registered: `legacy_sorting_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.sorting.BitonicSort` | `qualtran/bloqs/arithmetic/sorting.py` | Registered: `legacy_sorting_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.sorting.Comparator` | `qualtran/bloqs/arithmetic/sorting.py` | Registered: `legacy_sorting_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.sorting.ParallelComparators` | `qualtran/bloqs/arithmetic/sorting.py` | Registered: `legacy_sorting_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.subtraction.Subtract` | `qualtran/bloqs/arithmetic/subtraction.py` | Registered: `legacy_subtraction_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.subtraction.SubtractFrom` | `qualtran/bloqs/arithmetic/subtraction.py` | Registered: `legacy_subtraction_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.trigonometric.arcsin.ArcSin` | `qualtran/bloqs/arithmetic/trigonometric/arcsin.py` | Registered: `legacy_trigonometric_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.arithmetic.trigonometric.arctan.ArcTan` | `qualtran/bloqs/arithmetic/trigonometric/arctan.py` | Registered: `legacy_trigonometric_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.cnot.CNOT` | `qualtran/bloqs/basic_gates/cnot.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.global_phase.GlobalPhase` | `qualtran/bloqs/basic_gates/global_phase.py` | Registered: `legacy_scalar_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.hadamard.CHadamard` | `qualtran/bloqs/basic_gates/hadamard.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.hadamard.Hadamard` | `qualtran/bloqs/basic_gates/hadamard.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.identity.Identity` | `qualtran/bloqs/basic_gates/identity.py` | Registered: `legacy_exchange_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.on_each.OnEach` | `qualtran/bloqs/basic_gates/on_each.py` | Registered: `legacy_word_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.power.Power` | `qualtran/bloqs/basic_gates/power.py` | Registered: `legacy_power_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.rotation.CRz` | `qualtran/bloqs/basic_gates/rotation.py` | Registered: `legacy_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.rotation.CZPowGate` | `qualtran/bloqs/basic_gates/rotation.py` | Registered: `legacy_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.rotation.Rx` | `qualtran/bloqs/basic_gates/rotation.py` | Registered: `legacy_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.rotation.Ry` | `qualtran/bloqs/basic_gates/rotation.py` | Registered: `legacy_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.rotation.Rz` | `qualtran/bloqs/basic_gates/rotation.py` | Registered: `legacy_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.rotation.XPowGate` | `qualtran/bloqs/basic_gates/rotation.py` | Registered: `legacy_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.rotation.YPowGate` | `qualtran/bloqs/basic_gates/rotation.py` | Registered: `legacy_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.rotation.ZPowGate` | `qualtran/bloqs/basic_gates/rotation.py` | Registered: `legacy_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.s_gate.SGate` | `qualtran/bloqs/basic_gates/s_gate.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.su2_rotation.SU2RotationGate` | `qualtran/bloqs/basic_gates/su2_rotation.py` | Registered: `legacy_su2_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.swap.CSwap` | `qualtran/bloqs/basic_gates/swap.py` | Registered: `legacy_exchange_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.swap.Swap` | `qualtran/bloqs/basic_gates/swap.py` | Registered: `legacy_exchange_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.swap.TwoBitCSwap` | `qualtran/bloqs/basic_gates/swap.py` | Registered: `legacy_exchange_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.swap.TwoBitSwap` | `qualtran/bloqs/basic_gates/swap.py` | Registered: `legacy_exchange_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.t_gate.TGate` | `qualtran/bloqs/basic_gates/t_gate.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.toffoli.Toffoli` | `qualtran/bloqs/basic_gates/toffoli.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.x_basis.MinusEffect` | `qualtran/bloqs/basic_gates/x_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.x_basis.MinusState` | `qualtran/bloqs/basic_gates/x_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.x_basis.PlusEffect` | `qualtran/bloqs/basic_gates/x_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.x_basis.PlusState` | `qualtran/bloqs/basic_gates/x_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.x_basis.XGate` | `qualtran/bloqs/basic_gates/x_basis.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.y_gate.CYGate` | `qualtran/bloqs/basic_gates/y_gate.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.y_gate.YGate` | `qualtran/bloqs/basic_gates/y_gate.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.z_basis.CZ` | `qualtran/bloqs/basic_gates/z_basis.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.z_basis.IntEffect` | `qualtran/bloqs/basic_gates/z_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.z_basis.IntState` | `qualtran/bloqs/basic_gates/z_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.z_basis.OneEffect` | `qualtran/bloqs/basic_gates/z_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.z_basis.OneState` | `qualtran/bloqs/basic_gates/z_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.z_basis.ZGate` | `qualtran/bloqs/basic_gates/z_basis.py` | Registered: `legacy_chebyshev_factories.py, legacy_fixed_gate_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.z_basis.ZeroEffect` | `qualtran/bloqs/basic_gates/z_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.basic_gates.z_basis.ZeroState` | `qualtran/bloqs/basic_gates/z_basis.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.chebyshev_polynomial.ChebyshevPolynomial` | `qualtran/bloqs/block_encoding/chebyshev_polynomial.py` | Registered: `legacy_chebyshev_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.lcu_block_encoding.LCUBlockEncoding` | `qualtran/bloqs/block_encoding/lcu_block_encoding.py` | Registered: `legacy_block_encoding_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.lcu_block_encoding.SelectBlockEncoding` | `qualtran/bloqs/block_encoding/lcu_block_encoding.py` | Registered: `legacy_block_encoding_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.linear_combination.LinearCombination` | `qualtran/bloqs/block_encoding/linear_combination.py` | Registered: `legacy_linear_combination_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.phase.Phase` | `qualtran/bloqs/block_encoding/phase.py` | Registered: `legacy_phased_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.product.Product` | `qualtran/bloqs/block_encoding/product.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.sparse_matrix.ExplicitEntryOracle` | `qualtran/bloqs/block_encoding/sparse_matrix.py` | Registered: `legacy_sparse_matrix_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.sparse_matrix.SparseMatrix` | `qualtran/bloqs/block_encoding/sparse_matrix.py` | Registered: `legacy_sparse_matrix_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.sparse_matrix.TopLeftRowColumnOracle` | `qualtran/bloqs/block_encoding/sparse_matrix.py` | Registered: `legacy_sparse_matrix_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.sparse_matrix.UniformEntryOracle` | `qualtran/bloqs/block_encoding/sparse_matrix.py` | Registered: `legacy_sparse_matrix_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.tensor_product.TensorProduct` | `qualtran/bloqs/block_encoding/tensor_product.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.block_encoding.unitary.Unitary` | `qualtran/bloqs/block_encoding/unitary.py` | Registered: `legacy_chebyshev_factories.py, legacy_phased_block_factories.py, legacy_unitary_block_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.allocate.Allocate` | `qualtran/bloqs/bookkeeping/allocate.py` | Registered: `legacy_bookkeeping_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.always.Always` | `qualtran/bloqs/bookkeeping/always.py` | Registered: `legacy_bookkeeping_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.arbitrary_clifford.ArbitraryClifford` | `qualtran/bloqs/bookkeeping/arbitrary_clifford.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.auto_partition.AutoPartition` | `qualtran/bloqs/bookkeeping/auto_partition.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.cast.Cast` | `qualtran/bloqs/bookkeeping/cast.py` | Registered: `legacy_bookkeeping_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.free.Free` | `qualtran/bloqs/bookkeeping/free.py` | Registered: `legacy_bookkeeping_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.join.Join` | `qualtran/bloqs/bookkeeping/join.py` | Registered: `legacy_bookkeeping_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.partition.Join2` | `qualtran/bloqs/bookkeeping/partition.py` | Registered: `legacy_partition_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.partition.Partition` | `qualtran/bloqs/bookkeeping/partition.py` | Registered: `legacy_partition_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.partition.Split2` | `qualtran/bloqs/bookkeeping/partition.py` | Registered: `legacy_partition_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.bookkeeping.split.Split` | `qualtran/bloqs/bookkeeping/split.py` | Registered: `legacy_bookkeeping_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.black_boxes.ApplyControlledZs` | `qualtran/bloqs/chemistry/black_boxes.py` | Registered: `legacy_thc_selection_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.black_boxes.QROAM` | `qualtran/bloqs/chemistry/black_boxes.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.black_boxes.QROAMTwoRegs` | `qualtran/bloqs/chemistry/black_boxes.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.df.double_factorization.DoubleFactorizationBlockEncoding` | `qualtran/bloqs/chemistry/df/double_factorization.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.df.double_factorization.DoubleFactorizationOneBody` | `qualtran/bloqs/chemistry/df/double_factorization.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.df.prepare.InnerPrepareDoubleFactorization` | `qualtran/bloqs/chemistry/df/prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.df.prepare.OuterPrepareDoubleFactorization` | `qualtran/bloqs/chemistry/df/prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.df.prepare.OutputIndexedData` | `qualtran/bloqs/chemistry/df/prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.df.select_bloq.ProgRotGateArray` | `qualtran/bloqs/chemistry/df/select_bloq.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard.PrepareHubbard` | `qualtran/bloqs/chemistry/hubbard_model/qubitization/prepare_hubbard.py` | Registered: `legacy_hubbard_prepare_factories.py` | [PREPARE scoped closure](HUBBARD_PREPARATION_CONSTRUCTOR_AUDIT.md) |
| `qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard.HubbardMajorannaOperator` | `qualtran/bloqs/chemistry/hubbard_model/qubitization/select_hubbard.py` | Registered: `legacy_hubbard_selection_factories.py` | [Hubbard scoped closure](HUBBARD_SELECT_STORED_GRAPH_AUDIT.md) |
| `qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard.HubbardSpinUpZ` | `qualtran/bloqs/chemistry/hubbard_model/qubitization/select_hubbard.py` | Registered: `legacy_hubbard_selection_factories.py` | [Hubbard scoped closure](HUBBARD_SELECT_STORED_GRAPH_AUDIT.md) |
| `qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard.SelectHubbard` | `qualtran/bloqs/chemistry/hubbard_model/qubitization/select_hubbard.py` | Registered: `legacy_hubbard_selection_factories.py` | [Hubbard scoped closure](HUBBARD_SELECT_STORED_GRAPH_AUDIT.md) |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare.UniformSuperpostionIJFirstQuantization` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.FlagZeroAsFailure` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_nu.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.PrepareMuUnaryEncodedOneHot` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_nu.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.PrepareNuState` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_nu.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.PrepareNuSuperPositionState` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_nu.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.TestNuInequality` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_nu.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.TestNuLessThanMu` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_nu.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t.PreparePowerTwoState` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_t.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t.PrepareTFirstQuantization` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_t.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv.PrepareUVFirstQuantization` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_uv.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta.PrepareZetaState` | `qualtran/bloqs/chemistry/pbc/first_quantization/prepare_zeta.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu.PrepareMuUnaryEncodedOneHotWithProj` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/prepare_nu.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu.PrepareNuStateWithProj` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/prepare_nu.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t.PreparePowerTwoStateWithProj` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/prepare_t.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t.PrepareTFirstQuantizationWithProj` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/prepare_t.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_uv.PrepareUVFirstQuantizationWithProj` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/prepare_uv.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.ControlledMultiplexedCSwap3D` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/select_and_prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.PrepareFirstQuantizationWithProj` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/select_and_prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.PrepareTUVSuperpositions` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/select_and_prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.SelectFirstQuantizationWithProj` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/select_and_prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_t.SelectTFirstQuantizationWithProj` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/select_t.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_uv.SelectUVFirstQuantizationWithProj` | `qualtran/bloqs/chemistry/pbc/first_quantization/projectile/select_uv.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.MultiplexedCSwap3D` | `qualtran/bloqs/chemistry/pbc/first_quantization/select_and_prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.PrepareFirstQuantization` | `qualtran/bloqs/chemistry/pbc/first_quantization/select_and_prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.PrepareTUVSuperpositions` | `qualtran/bloqs/chemistry/pbc/first_quantization/select_and_prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.SelectFirstQuantization` | `qualtran/bloqs/chemistry/pbc/first_quantization/select_and_prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.UniformSuperpostionIJFirstQuantization` | `qualtran/bloqs/chemistry/pbc/first_quantization/select_and_prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.select_t.SelectTFirstQuantization` | `qualtran/bloqs/chemistry/pbc/first_quantization/select_t.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.select_uv.ApplyNuclearPhase` | `qualtran/bloqs/chemistry/pbc/first_quantization/select_uv.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.pbc.first_quantization.select_uv.SelectUVFirstQuantization` | `qualtran/bloqs/chemistry/pbc/first_quantization/select_uv.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.sf.prepare.InnerPrepareSingleFactorization` | `qualtran/bloqs/chemistry/sf/prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.sf.prepare.OuterPrepareSingleFactorization` | `qualtran/bloqs/chemistry/sf/prepare.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.sf.select_bloq.SelectSingleFactorization` | `qualtran/bloqs/chemistry/sf/select_bloq.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.sf.single_factorization.SingleFactorizationBlockEncoding` | `qualtran/bloqs/chemistry/sf/single_factorization.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.sf.single_factorization.SingleFactorizationOneBody` | `qualtran/bloqs/chemistry/sf/single_factorization.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.sparse.prepare.PrepareSparse` | `qualtran/bloqs/chemistry/sparse/prepare.py` | Registered: `legacy_sparse_preparation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.sparse.select_bloq.SelectSparse` | `qualtran/bloqs/chemistry/sparse/select_bloq.py` | Registered: `legacy_sparse_chemistry_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.thc.prepare.PrepareTHC` | `qualtran/bloqs/chemistry/thc/prepare.py` | Registered: `legacy_thc_preparation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.thc.prepare.UniformSuperpositionTHC` | `qualtran/bloqs/chemistry/thc/prepare.py` | Registered: `legacy_thc_preparation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.thc.select_bloq.SelectTHC` | `qualtran/bloqs/chemistry/thc/select_bloq.py` | Registered: `legacy_thc_selection_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.thc.select_bloq.THCRotations` | `qualtran/bloqs/chemistry/thc/select_bloq.py` | Registered: `legacy_thc_selection_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt.NewtonRaphsonApproxInverseSquareRoot` | `qualtran/bloqs/chemistry/trotter/grid_ham/inverse_sqrt.py` | Registered: `legacy_grid_inverse_sqrt_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt.PolynmomialEvaluationInverseSquareRoot` | `qualtran/bloqs/chemistry/trotter/grid_ham/inverse_sqrt.py` | Registered: `legacy_grid_inverse_sqrt_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.grid_ham.kinetic.KineticEnergy` | `qualtran/bloqs/chemistry/trotter/grid_ham/kinetic.py` | Registered: `legacy_grid_kinetic_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.grid_ham.potential.PairPotential` | `qualtran/bloqs/chemistry/trotter/grid_ham/potential.py` | Registered: `legacy_grid_potential_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.grid_ham.potential.PotentialEnergy` | `qualtran/bloqs/chemistry/trotter/grid_ham/potential.py` | Registered: `legacy_grid_potential_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.grid_ham.qvr.QuantumVariableRotation` | `qualtran/bloqs/chemistry/trotter/grid_ham/qvr.py` | Registered: `legacy_grid_qvr_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.hubbard.hopping.HoppingPlaquette` | `qualtran/bloqs/chemistry/trotter/hubbard/hopping.py` | Registered: `legacy_hubbard_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.hubbard.hopping.HoppingTile` | `qualtran/bloqs/chemistry/trotter/hubbard/hopping.py` | Registered: `legacy_hubbard_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.hubbard.hopping.HoppingTileHWP` | `qualtran/bloqs/chemistry/trotter/hubbard/hopping.py` | Registered: `legacy_hubbard_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.hubbard.interaction.Interaction` | `qualtran/bloqs/chemistry/trotter/hubbard/interaction.py` | Registered: `legacy_hubbard_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.hubbard.interaction.InteractionHWP` | `qualtran/bloqs/chemistry/trotter/hubbard/interaction.py` | Registered: `legacy_hubbard_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.chemistry.trotter.ising.unitaries.IsingXUnitary` | `qualtran/bloqs/chemistry/trotter/ising/unitaries.py` | Registered: `legacy_ising_factories.py` | [Ising source closure](ISING_TROTTER_FILE_AUDIT.md) |
| `qualtran.bloqs.chemistry.trotter.ising.unitaries.IsingZZUnitary` | `qualtran/bloqs/chemistry/trotter/ising/unitaries.py` | Registered: `legacy_ising_factories.py` | [Ising source closure](ISING_TROTTER_FILE_AUDIT.md) |
| `qualtran.bloqs.chemistry.trotter.trotterized_unitary` | `qualtran/bloqs/chemistry/trotter/trotterized_unitary.py` | Module-valued entry | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography._factoring_shims.MeasureQFT` | `qualtran/bloqs/cryptography/_factoring_shims.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.ecc.ec_add.ECAdd` | `qualtran/bloqs/cryptography/ecc/ec_add.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepFive` | `qualtran/bloqs/cryptography/ecc/ec_add.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepFour` | `qualtran/bloqs/cryptography/ecc/ec_add.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepOne` | `qualtran/bloqs/cryptography/ecc/ec_add.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepSix` | `qualtran/bloqs/cryptography/ecc/ec_add.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepThree` | `qualtran/bloqs/cryptography/ecc/ec_add.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepTwo` | `qualtran/bloqs/cryptography/ecc/ec_add.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.ecc.ec_add_r.ECWindowAddR` | `qualtran/bloqs/cryptography/ecc/ec_add_r.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.rsa.rsa_mod_exp.ModExp` | `qualtran/bloqs/cryptography/rsa/rsa_mod_exp.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.cryptography.rsa.rsa_phase_estimate.RSAPhaseEstimate` | `qualtran/bloqs/cryptography/rsa/rsa_phase_estimate.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.data_loading.qroam_clean.QROAMClean` | `qualtran/bloqs/data_loading/qroam_clean.py` | Registered: `legacy_qroam_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.data_loading.qroam_clean.QROAMCleanAdjoint` | `qualtran/bloqs/data_loading/qroam_clean.py` | Registered: `legacy_qroam_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.data_loading.qroam_clean.QROAMCleanAdjointWrapper` | `qualtran/bloqs/data_loading/qroam_clean.py` | Registered: `legacy_qroam_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.data_loading.qrom.QROM` | `qualtran/bloqs/data_loading/qrom.py` | Registered: `legacy_qrom_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.data_loading.select_swap_qrom.SelectSwapQROM` | `qualtran/bloqs/data_loading/select_swap_qrom.py` | Registered: `legacy_select_swap_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.atom.TestAtom` | `qualtran/bloqs/for_testing/atom.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.atom.TestGWRAtom` | `qualtran/bloqs/for_testing/atom.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.atom.TestTwoBitOp` | `qualtran/bloqs/for_testing/atom.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.casting.TestCastToFrom` | `qualtran/bloqs/for_testing/casting.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.interior_alloc.InteriorAlloc` | `qualtran/bloqs/for_testing/interior_alloc.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.many_registers.TestBoundedQUInt` | `qualtran/bloqs/for_testing/many_registers.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.many_registers.TestMultiRegister` | `qualtran/bloqs/for_testing/many_registers.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.many_registers.TestMultiTypedRegister` | `qualtran/bloqs/for_testing/many_registers.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.many_registers.TestQFxp` | `qualtran/bloqs/for_testing/many_registers.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.matrix_gate.MatrixGate` | `qualtran/bloqs/for_testing/matrix_gate.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.with_call_graph.TestBloqWithCallGraph` | `qualtran/bloqs/for_testing/with_call_graph.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.with_decomposition.TestIndependentParallelCombo` | `qualtran/bloqs/for_testing/with_decomposition.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.with_decomposition.TestParallelCombo` | `qualtran/bloqs/for_testing/with_decomposition.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.for_testing.with_decomposition.TestSerialCombo` | `qualtran/bloqs/for_testing/with_decomposition.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.hamiltonian_simulation.guided_hamiltonian.GuidedHamiltonian` | `qualtran/bloqs/hamiltonian_simulation/guided_hamiltonian.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.hamiltonian_simulation.guided_hamiltonian.GuidedHamiltonianPhaseEstimation` | `qualtran/bloqs/hamiltonian_simulation/guided_hamiltonian.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp.HamiltonianSimulationByGQSP` | `qualtran/bloqs/hamiltonian_simulation/hamiltonian_simulation_by_gqsp.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.mcmt.and_bloq.And` | `qualtran/bloqs/mcmt/and_bloq.py` | Registered: `legacy_conjunction_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mcmt.and_bloq.MultiAnd` | `qualtran/bloqs/mcmt/and_bloq.py` | Registered: `legacy_conjunction_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd` | `qualtran/bloqs/mcmt/controlled_via_and.py` | Registered: `legacy_chebyshev_factories.py, legacy_controlled_factories.py, legacy_linear_combination_factories.py, legacy_multi_control_factories.py, legacy_phased_block_factories.py, legacy_qubitization_walk_factories.py, legacy_sparse_chemistry_factories.py, legacy_sparse_matrix_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mcmt.ctrl_spec_and.CtrlSpecAnd` | `qualtran/bloqs/mcmt/ctrl_spec_and.py` | Registered: `legacy_predicate_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mcmt.multi_control_pauli.MultiControlPauli` | `qualtran/bloqs/mcmt/multi_control_pauli.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.mcmt.multi_control_pauli.MultiControlX` | `qualtran/bloqs/mcmt/multi_control_pauli.py` | Registered: `legacy_multi_control_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mcmt.multi_control_pauli.MultiControlZ` | `qualtran/bloqs/mcmt/multi_control_pauli.py` | Registered: `legacy_multi_control_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mcmt.multi_target_cnot.MultiTargetCNOT` | `qualtran/bloqs/mcmt/multi_target_cnot.py` | Registered: `legacy_fanout_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mean_estimation.complex_phase_oracle.ComplexPhaseOracle` | `qualtran/bloqs/mean_estimation/complex_phase_oracle.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.mean_estimation.mean_estimation_operator.MeanEstimationOperator` | `qualtran/bloqs/mean_estimation/mean_estimation_operator.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.CModAddK` | `qualtran/bloqs/mod_arithmetic/mod_addition.py` | Registered: `legacy_modular_addition_factories.py` | Exact alias wire scope: [audit](MODULAR_PACKAGE_EXPORT_AUDIT.md); full module boundaries retained |
| `qualtran.bloqs.mod_arithmetic.CModSub` | `qualtran/bloqs/mod_arithmetic/mod_subtraction.py` | Registered: `legacy_modular_subtraction_factories.py` | Exact alias wire scope: [audit](MODULAR_PACKAGE_EXPORT_AUDIT.md); full module boundaries retained |
| `qualtran.bloqs.mod_arithmetic.ModAdd` | `qualtran/bloqs/mod_arithmetic/mod_addition.py` | Registered: `legacy_modular_addition_factories.py` | Exact alias wire scope: [audit](MODULAR_PACKAGE_EXPORT_AUDIT.md); full module boundaries retained |
| `qualtran.bloqs.mod_arithmetic.ModSub` | `qualtran/bloqs/mod_arithmetic/mod_subtraction.py` | Registered: `legacy_modular_subtraction_factories.py` | Exact alias wire scope: [audit](MODULAR_PACKAGE_EXPORT_AUDIT.md); full module boundaries retained |
| `qualtran.bloqs.mod_arithmetic.mod_addition.CModAdd` | `qualtran/bloqs/mod_arithmetic/mod_addition.py` | Registered: `legacy_modular_addition_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_addition.CModAddK` | `qualtran/bloqs/mod_arithmetic/mod_addition.py` | Registered: `legacy_modular_addition_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_addition.CtrlScaleModAdd` | `qualtran/bloqs/mod_arithmetic/mod_addition.py` | Registered: `legacy_modular_addition_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_addition.ModAdd` | `qualtran/bloqs/mod_arithmetic/mod_addition.py` | Registered: `legacy_modular_addition_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_addition.ModAddK` | `qualtran/bloqs/mod_arithmetic/mod_addition.py` | Registered: `legacy_modular_addition_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_division.KaliskiModInverse` | `qualtran/bloqs/mod_arithmetic/mod_division.py` | Registered: `legacy_modular_inverse_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_division._KaliskiIteration` | `qualtran/bloqs/mod_arithmetic/mod_division.py` | Registered: `legacy_modular_inverse_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_multiplication.CModMulK` | `qualtran/bloqs/mod_arithmetic/mod_multiplication.py` | Registered: `legacy_modular_multiplication_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_multiplication.DirtyOutOfPlaceMontgomeryModMul` | `qualtran/bloqs/mod_arithmetic/mod_multiplication.py` | Registered: `legacy_modular_multiplication_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_multiplication.ModDbl` | `qualtran/bloqs/mod_arithmetic/mod_multiplication.py` | Registered: `legacy_modular_multiplication_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_multiplication.SingleWindowModMul` | `qualtran/bloqs/mod_arithmetic/mod_multiplication.py` | Registered: `legacy_modular_multiplication_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_subtraction.CModNeg` | `qualtran/bloqs/mod_arithmetic/mod_subtraction.py` | Registered: `legacy_modular_subtraction_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.mod_arithmetic.mod_subtraction.ModNeg` | `qualtran/bloqs/mod_arithmetic/mod_subtraction.py` | Registered: `legacy_modular_subtraction_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.multiplexers.apply_gate_to_lth_target.ApplyGateToLthQubit` | `qualtran/bloqs/multiplexers/apply_gate_to_lth_target.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.multiplexers.apply_lth_bloq.ApplyLthBloq` | `qualtran/bloqs/multiplexers/apply_lth_bloq.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.multiplexers.black_box_select.BlackBoxSelect` | `qualtran/bloqs/multiplexers/black_box_select.py` | Registered: `legacy_packed_oracle_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.multiplexers.select_base.SelectOracle` | `qualtran/bloqs/multiplexers/select_base.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.multiplexers.select_pauli_lcu.SelectPauliLCU` | `qualtran/bloqs/multiplexers/select_pauli_lcu.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.multiplexers.selected_majorana_fermion.SelectedMajoranaFermion` | `qualtran/bloqs/multiplexers/selected_majorana_fermion.py` | Registered: `legacy_majorana_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.multiplexers.unary_iteration_bloq.UnaryIterationGate` | `qualtran/bloqs/multiplexers/unary_iteration_bloq.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_adjacency_list.ColumnOfKthNonZeroEntry` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_adjacency_list.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_adjacency_list.IndexOfNonZeroColumn` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_adjacency_list.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_adjacency_list.KikuchiNonZeroIndex` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_adjacency_list.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_adjacency_matrix.KikuchiMatrixEntry` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_adjacency_matrix.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_block_encoding.BlackBoxKikuchiEntryOracle` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_block_encoding.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_block_encoding.BlackBoxKikuchiRowColumnOracle` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_block_encoding.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_block_encoding.KikuchiHamiltonian` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_block_encoding.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state.GuidingState` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_guiding_state.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state.ProbabilisticUncompute` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_guiding_state.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state.SimpleGuidingState` | `qualtran/bloqs/optimization/k_xor_sat/kikuchi_guiding_state.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.load_kxor_instance.LoadConstraintScopes` | `qualtran/bloqs/optimization/k_xor_sat/load_kxor_instance.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.load_kxor_instance.LoadUniqueScopeIndex` | `qualtran/bloqs/optimization/k_xor_sat/load_kxor_instance.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.optimization.k_xor_sat.load_kxor_instance.PRGAUniqueConstraintRHS` | `qualtran/bloqs/optimization/k_xor_sat/load_kxor_instance.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.phase_estimation.kaiser_window_state.KaiserWindowState` | `qualtran/bloqs/phase_estimation/kaiser_window_state.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.phase_estimation.lp_resource_state.LPRSInterimPrep` | `qualtran/bloqs/phase_estimation/lp_resource_state.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.phase_estimation.lp_resource_state.LPResourceState` | `qualtran/bloqs/phase_estimation/lp_resource_state.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.phase_estimation.qpe_window_state.RectangularWindowState` | `qualtran/bloqs/phase_estimation/qpe_window_state.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.phase_estimation.qubitization_qpe.QubitizationQPE` | `qualtran/bloqs/phase_estimation/qubitization_qpe.py` | Registered: `legacy_qubitization_qpe_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.phase_estimation.text_book_qpe.TextbookQPE` | `qualtran/bloqs/phase_estimation/text_book_qpe.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.qft.approximate_qft.ApproximateQFT` | `qualtran/bloqs/qft/approximate_qft.py` | Registered: `legacy_approximate_qft_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.qft.qft_phase_gradient.QFTPhaseGradient` | `qualtran/bloqs/qft/qft_phase_gradient.py` | Registered: `legacy_gradient_qft_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.qft.qft_text_book.QFTTextBook` | `qualtran/bloqs/qft/qft_text_book.py` | Registered: `legacy_qft_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.qft.two_bit_ffft.TwoBitFFFT` | `qualtran/bloqs/qft/two_bit_ffft.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.qsp.generalized_qsp.GeneralizedQSP` | `qualtran/bloqs/qsp/generalized_qsp.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.qubitization.qubitization_walk_operator.QubitizationWalkOperator` | `qualtran/bloqs/qubitization/qubitization_walk_operator.py` | Registered: `legacy_qubitization_walk_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.reflections.prepare_identity.PrepareIdentity` | `qualtran/bloqs/reflections/prepare_identity.py` | Registered: `legacy_alias_preparation_factories.py, legacy_reflection_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.reflections.reflection_using_prepare.ReflectionUsingPrepare` | `qualtran/bloqs/reflections/reflection_using_prepare.py` | Registered: `legacy_alias_preparation_factories.py, legacy_reflection_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.hamming_weight_phasing.HammingWeightPhasing` | `qualtran/bloqs/rotations/hamming_weight_phasing.py` | Registered: `legacy_hamming_phase_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.hamming_weight_phasing.HammingWeightPhasingViaPhaseGradient` | `qualtran/bloqs/rotations/hamming_weight_phasing.py` | Registered: `legacy_hamming_phase_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.phase_gradient.AddIntoPhaseGrad` | `qualtran/bloqs/rotations/phase_gradient.py` | Registered: `legacy_phase_gradient_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.phase_gradient.AddScaledValIntoPhaseReg` | `qualtran/bloqs/rotations/phase_gradient.py` | Registered: `legacy_phase_gradient_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.phase_gradient.PhaseGradientState` | `qualtran/bloqs/rotations/phase_gradient.py` | Registered: `legacy_phase_gradient_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.phase_gradient.PhaseGradientUnitary` | `qualtran/bloqs/rotations/phase_gradient.py` | Registered: `legacy_phase_gradient_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.phasing_via_cost_function.PhasingViaCostFunction` | `qualtran/bloqs/rotations/phasing_via_cost_function.py` | Registered: `legacy_cost_phasing_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.programmable_ancilla_rotation.ZPowProgrammedAncilla` | `qualtran/bloqs/rotations/programmable_ancilla_rotation.py` | Registered: `legacy_programmed_phase_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.programmable_ancilla_rotation.ZPowUsingProgrammedAncilla` | `qualtran/bloqs/rotations/programmable_ancilla_rotation.py` | Registered: `legacy_programmed_phase_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.programmable_rotation_gate_array.ProgrammableRotationGateArray` | `qualtran/bloqs/rotations/programmable_rotation_gate_array.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.programmable_rotation_gate_array.ProgrammableRotationGateArrayBase` | `qualtran/bloqs/rotations/programmable_rotation_gate_array.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.quantum_variable_rotation.QvrInterface` | `qualtran/bloqs/rotations/quantum_variable_rotation.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.quantum_variable_rotation.QvrPhaseGradient` | `qualtran/bloqs/rotations/quantum_variable_rotation.py` | Registered: `legacy_cost_phasing_factories.py, legacy_qvr_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.quantum_variable_rotation.QvrZPow` | `qualtran/bloqs/rotations/quantum_variable_rotation.py` | Registered: `legacy_cost_phasing_factories.py, legacy_qvr_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.rz_via_phase_gradient.RzViaPhaseGradient` | `qualtran/bloqs/rotations/rz_via_phase_gradient.py` | Registered: `legacy_phase_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.rotations.zpow_via_phase_gradient.ZPowConstViaPhaseGradient` | `qualtran/bloqs/rotations/zpow_via_phase_gradient.py` | Registered: `legacy_phase_rotation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.state_preparation.black_box_prepare.BlackBoxPrepare` | `qualtran/bloqs/state_preparation/black_box_prepare.py` | Registered: `legacy_packed_oracle_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.state_preparation.prepare_base.PrepareOracle` | `qualtran/bloqs/state_preparation/prepare_base.py` | Unregistered | Published module receipts pending reconciliation |
| `qualtran.bloqs.state_preparation.prepare_uniform_superposition.PrepareUniformSuperposition` | `qualtran/bloqs/state_preparation/prepare_uniform_superposition.py` | Registered: `legacy_uniform_preparation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.state_preparation.state_preparation_alias_sampling.SparseStatePreparationAliasSampling` | `qualtran/bloqs/state_preparation/state_preparation_alias_sampling.py` | Registered: `legacy_alias_preparation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.state_preparation.state_preparation_alias_sampling.StatePreparationAliasSampling` | `qualtran/bloqs/state_preparation/state_preparation_alias_sampling.py` | Registered: `legacy_alias_preparation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.state_preparation.state_preparation_via_rotation.PRGAViaPhaseGradient` | `qualtran/bloqs/state_preparation/state_preparation_via_rotation.py` | Registered: `legacy_rotation_preparation_factories.py, legacy_sparse_rotation_preparation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.state_preparation.state_preparation_via_rotation.StatePreparationViaRotations` | `qualtran/bloqs/state_preparation/state_preparation_via_rotation.py` | Registered: `legacy_rotation_preparation_factories.py, legacy_sparse_rotation_preparation_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.swap_network.cswap_approx.CSwapApprox` | `qualtran/bloqs/swap_network/cswap_approx.py` | Registered: `legacy_relative_swap_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.swap_network.multiplexed_cswap.MultiplexedCSwap` | `qualtran/bloqs/swap_network/multiplexed_cswap.py` | Registered: `legacy_swap_network_factories.py` | Published module receipts pending reconciliation |
| `qualtran.bloqs.swap_network.swap_with_zero.SwapWithZero` | `qualtran/bloqs/swap_network/swap_with_zero.py` | Registered: `legacy_swap_network_factories.py` | Published module receipts pending reconciliation |
| `qualtran.cirq_interop._cirq_to_bloq.CirqGateAsBloq` | `qualtran/cirq_interop/_cirq_to_bloq.py` | Unregistered | Published module receipts pending reconciliation |
