# Quantum kernel workflow

The synthetic generator uses the source centres, the per-feature seeded normal
draws, clipping, and the original normalization. Iris retains every versicolor
and virginica row and both petal measurements. The source fits the scalers before
its two stratified splits. The walkthrough preserves that order and seed 42.

The native Bloch map uses the existing RX and RZ helper. The Pauli map applies
two layers of Hadamards and the commuting Z, Z, ZZ evolutions. The coefficients
are the two original features and the product of their offsets from pi.
The overlap circuits compose each map with the other map's adjoint and read
the all-zero outcome. Independent small Circuits check both maps, the complete
complex overlap states, and the all-zero probabilities.

The training workflow executes the native feature states once per point.
An inner product of two such states gives the exact all-zero event probability
for their overlap circuit. The host draws 2,048 Bernoulli trials per queried
kernel entry from that probability, using seeded binomial counts. It computes
one triangle of the training matrix and uses only the selected support vectors
for test and prediction entries. No PSD repair or prediction correction is
applied to the sampled matrix. The original uses a hosted sampler and does not
set the shot budget in the Notebook. The local shot budget and seed are explicit
adaptations. Test and prediction scores come from the executed classifiers.

The classical baseline uses the original RBF SVC and the same Iris splits.
The example's scikit-learn installation belongs to its local Notebook
environment.
