# Source-to-native workflow mapping

The pinned source is `tutorials/basic_tutorials/qmci_pi_estimation/qmci_pi_estimation.ipynb`
at commit `b69c4893d8264d1e59ce0909df2bd88efed25e00`. Provenance records every
reviewed source code cell and the original file and Git blob hashes.

| Source cells | Native behavior |
| --- | --- |
| 6 | Four-bit lattice plot with all 256 points and strict circle boundary |
| 8 | Classical grid axis lengths 3, 5 and 6; these are lengths, not bit widths |
| 10, 11 | Original three-bit QPE, six retained coordinate bits, 2048 actual local observations |
| 12 | Original sine reporting formula, separately labeled squared probability and continuum pi |
| 15, 16 | Count-only bounded iterative profile and three-bit preparation resources |
| 18, 19, 20 | Independent three-bit run, epsilon .03, alpha .01, counts, interval and pi estimate |
| 23 | Fresh runs at all four source coordinate precisions, full-state checks for every visited power |
| 25 | Actual local estimates replace commented illustrative hosted outputs |
| 26 | All sixteen classical grid sizes, quantum error bars, logarithmic lattice-size axis and true-pi line |

The native root uses checked `allocate(bits, q)` and separate coordinate fields.
This preserves the source physical data without introducing bound type aliases.
The existing public signatures, exports, literal annotations and quantum-only
counterpart remain unchanged. Every visited native circuit has an independent
complex128 Circuit comparison, absolute global phase check and zero-work check.
QPE retains dropped coordinates in the comparison. Dense references are bounded
to the original small examples and are never used to generate native programs.

The example estimator receives only success counts. Exact finite-grid answers
and analytic amplification probabilities appear only in tests. Seeds 1801..1804
identify the four sweep runs; seed 1793 identifies the separate three-bit run;
seed 1800 identifies QPE. Repeated bins sample fresh counts from a cached actual
native marginal. This is a local simulation profile, not hardware timing.

Statistical confidence concerns the finite-grid probability. Finite-grid bias,
three-bit phase resolution, the source sine reporting formula and statistical
sampling error remain separate. No claim of matching hosted samples, Classiq
IQAE implementation, asymptotic speedup or optimal gate resources is made.
