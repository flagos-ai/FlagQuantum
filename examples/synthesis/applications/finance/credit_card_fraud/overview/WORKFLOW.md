# Credit-card data and kernel checks

The source asset provides 948 transactions and 29 retained feature columns.
The workflow separates the rows by label and preserves within-class order for
the 125/60/60 partitions. PCA and standardization fit the training data. The
source's final angular scaling fits the union of the three partitions and maps
features to `[-pi, pi]`. Each t-SNE projection uses the source's seed and its
own raw split.

The native Pauli map applies two layers, three Z terms, and all three ZZ pairs.
The reference compares the complete overlap Circuit using two pairs of freshly
transformed points. Sampling reuses executed feature states and their exact
all-zero overlap probabilities. The host draws local binomial counts with
2,048 trials per queried entry, trains the sampled-kernel SVC, and evaluates
test and prediction entries against its actual support vectors. The
Classiq example leaves the hosted shot budget implicit, so the local budget
and seed are recorded explicitly.

The workflow also runs the linear, polynomial, RBF and sigmoid SVC baselines on
the same transformed training and test rows. The reported figures describe
those specific source splits and the seeded local experiment.
