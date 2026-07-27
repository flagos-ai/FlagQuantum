# Smoke Results

This directory contains development, environment, rank-local, or collective
smoke results. These files may exercise sharded layouts or collectives, but they
are not promoted scalability evidence.

Smoke payloads must use `claim_evidence_type="development_smoke"` when they
look distributed, keep `scalability_claim_allowed=false`, and include explicit
release blockers explaining why they cannot be promoted.
