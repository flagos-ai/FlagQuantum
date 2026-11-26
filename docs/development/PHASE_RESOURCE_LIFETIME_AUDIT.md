# Complex phase resource lifetimes

The frozen Qualtran v0.7.0 reference is commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, specifically
`bloqs/mean_estimation/complex_phase_oracle.py`.

The original schedule releases the sign and magnitude outputs before calling
the encoder adjoint. Consequently its caller-supplied peak does not retain those
phase registers as bystanders. The native model previously added them twice for
that stage. This correction changes resource estimates only; emitted primitive
instructions, measurement feedback and execution remain unchanged.

Independent source encoder models with forward/inverse peaks (4,7), (11,4) and
(2,2) produce complete phase peaks 7,11 and5 respectively, for both directions.
`test_complex_phase_resource_lifetimes.py` checks every gate category and the
peak against unmodified source cost queries. These are explicit static child
models, not measured primitive inventories.

The complete parent acceptance and remaining execution boundaries remain in
[the phase file audit](COMPLEX_PHASE_ORACLE_FILE_AUDIT.md).
