# Physical packed spin benchmark

K-0042 and K-0043 track the locked `main_spin.cc` and its original tests
(blobs `e3afa60b159983e9f1a54b5bf93055a0586b3430` and
`b0a449527f8a5d22ac26fbb8e9626e7f43f79aea`).

Run `python -m flagquantum.ecosystem.kmx_spin --mode 256 --shots 1024 --in circuit.kmb`.
The 64-, 128-, 256- and 512-lane modes also accept the `-polyfill`
spelling. File options are optional for piping. The portable executor
uses Python packed integers for every mode. Its word order follows
the locked scalar/polyfill layout measured on this host. The source
C++ SIMD initialization on other hosts can depend on compiler argument
evaluation order and has no independent parity claim here.

Simulation seeds its vertical ChaCha8 stream with the first eleven
packed Mersenne Twister words. The remaining words initialize the
sign, quantum wires, and classical bits, in that order, including sparse
wire holes. Each requested batch runs a full packed scan and clears
the guard stack. Its quantum, classical, sign, angle, and random-stream
state carry to the next scan. Every HMR and R consumes a mask, including
under a false guard. The hash folds sign and all quantum and classical
wire values by XOR, listing lane zero first. Exact rotation angles are
retained through simulation and excluded from the source hash.

The command reports physical record count multiplied by requested
shots, even for the final partial batch. This nominal benchmark counter
includes tags, debug records and conditions, matching the source.
The elapsed and throughput columns use actual local microsecond timing
and source SI formatting. Native timing does not stand for SIMD
performance.

## Verification

Both original spin tests pass against the unchanged locked sources.
Eighty locked reports match the deterministic hash and record-count
lines with KMX and KMB inputs. They cover all eight mode spellings,
empty, incrementer and measured inputs, and a sparse mixed case that
crosses both a Mersenne Twister state boundary and the ChaCha8 refill.
The controlled measurements, quantum and classical lanes, and exact
rotations run through the same physical executor used for sampling.
A forced two-scan tape checks retained sign, exact angles, feedback,
and work cleanup. The shared sampler's complex-amplitude cross-check
remains in the regression. Pipe and output file behavior are covered.
The native command rejects missing or invalid arguments before it
writes the destination.
