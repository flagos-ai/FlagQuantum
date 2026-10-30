# Ascend 910C hardware development evidence

These compact reports support the current
[Ascend capability record](../../ascend_910c_20261030.json) and its
[installation and validation guide](../../../docs/guides/ASCEND_910C.md).
They retain per-rank numerical comparisons, gradient ownership, checkpoint
restart results, operator tests and the regression that motivated the FP64
communication synchronization fix. They establish the documented bounded
functional scope; they are not performance or capacity evidence.

The capability record identifies each producing source, dependency and wheel,
with digests for both raw and public files. Public copies redact infrastructure
identifiers and home paths. Text logs are gzip-compressed, and JUnit XML hostnames
are redacted. Numerical results are preserved. The F3 open environment metadata
adds an explicit external Torch-FL source-origin annotation.

Full raw archives, checkpoints, wheels and intermediate build trees remain in
the task evidence archive rather than this source repository. Earlier failed
and passing records are retained separately; the final combined hardware run
is `node1/full-fq-final-v75.log.gz` with its per-case JUnit XML.
