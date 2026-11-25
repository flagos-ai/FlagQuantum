# Local state preparation workflow

The reviewed upstream Notebook contains exactly two code cells: a quantum
function definition and a synthesis call. Both are implemented by the native
`main.py`; there is no training loop, external data, plotting or remote sampling
session to migrate. The script additionally executes the program and reads its
named output probabilities.

The native `QNum` output is a whole-register view of the upstream `QArray[QBit]`.
The program performs no element indexing; its original width and state are
unchanged. No source importer runs in the user example.

The workflow acceptance test runs the actual script. The independent Circuit
comparison checks full complex amplitudes, absolute phase, register order and
zero hidden workspace. The [provenance record](provenance.json) pins both the
original quantum source and the reviewed Notebook code cells.
