# Comparison Results

This directory contains cross-framework or backend comparison results. These
files can compare runtime, device, interface, dtype, gradient method, and other
benchmark settings, but they are not release-grade scalability evidence.

Comparison payloads must keep `scalability_claim_allowed=false` and include a
non-release blocker such as `comparison_result_not_release_scalability_evidence`.
If a comparison used multiple ranks or devices without proving one logical
workload capacity expansion, it is labeled as rank-local or non-release rather
than scalability.
