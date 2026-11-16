# Symmetric-weight DQI for Max-XORSAT

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.main
python -m examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.comparisons.circuit
```

The native program prepares six error bits in a symmetric superposition of
weight-zero, weight-one and weight-two words. Each word of weight k receives
weight[k]/sqrt(comb(6,k)). The retained fourth amplitude is zero; unsupported
weight-three support and unnormalized or nonfinite weights are rejected.
This is an input-state construction, not a precomputed final solution.

The program retains all six GF(2) columns, parity signs and 22 decoder entries.
The decoder XORs into existing error storage under coherent syndrome controls;
static X masks are restored after each lookup. Both six-bit words remain visible.
The source's consumed binary/unary preparation storage is never allocated here;
this example does not introduce a native free operation.

Independent acceptance compares all 4,096 complex amplitudes against a Circuit
using a Householder state preparation and explicit decoder permutation matrices.
Three normalized weight vectors include negative amplitudes. Additional decoder
checks use nonuniform complex targets and all syndrome values, test inverse
restoration, alias refusal and budgets. All clean work must return to zero.

The native default uses 15 physical bits including 3 clean work bits, 403
operations and depth 352. The frozen original uses 14 physical bits including
2 work bits, 3,802 operations and depth 1,988. These are local CPU resource
counts, not performance or complete Notebook-workflow certification.
