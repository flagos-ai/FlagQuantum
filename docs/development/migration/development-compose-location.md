# Development Compose location

Integration declaration, before implementation. Owner: Integration.

The development Compose file belongs in `docker/compose.dev.yaml`. Its content
and four service definitions remain unchanged. Commands run from the repository
root and explicitly select that root as the Compose project directory:

```bash
docker compose --project-directory . -f docker/compose.dev.yaml --profile cpu config
```

The command prints the resolved CPU development configuration without building,
pulling or starting a container. Existing development commands keep their
profile, service and arguments; add the project-directory option and use the new
file location. This preserves the build context, Dockerfile, checkout mount,
root `.env` discovery and default checkout-derived project name. An explicit
`COMPOSE_PROJECT_NAME` continues to take precedence. No fixed project name,
new launcher, old-path copy or compatibility forwarding is introduced.

## Independent source and consumer closure

The qualified prerequisite is PR #1355, head
`cd3a8a0496167923de2aeab29a431434f1c12a39`. The new branch carries its exact tree
`274f484d481004bdeeb94bbef52fe994e20c72b3` on main
`29afa414a6362d381cb4c2992cd19fd024761f83`; predecessor checks do not qualify this
new head.

Incoming PR #1084 at `f765ed390540ece25c3e47df227565cd595d4542` retains exactly
the same Compose, development README and Dockerfile blobs. Its publication
workflow is the unchanged old-base version, while main has already evolved that
workflow. Preserve main's complete workflow and update only the two Compose
path filters. The current four reference files are that workflow, the ownership
policy, the development README and the retained-family migration document.
An exact changed-blob audit covered all 4,677 incoming changed blobs, including
4,666 UTF-8 text blobs, and found no changed incoming reference to the Compose
filename. This is independence evidence for this file/consumer closure, not
whole-PR or merged architecture acceptance.

The earlier decision to defer Compose is superseded by this measured closure:
the incoming workflow reference already exists in main and does not require
the incoming program/type implementation. The eventual incoming reconciliation
must retain this relocation and the evolved main workflow.

## Implementation and acceptance

1. Move the unchanged YAML; remove its root predecessor.
2. Update both workflow filters and the Integration-owned policy path.
3. Update every development command with the explicit project directory, plus
   the retained-family explanation and migration ledger/path map.
4. Compare complete resolved configurations with the original for all four
   profiles and all profiles, with defaults, controlled root `.env` values and
   shell/project-name overrides, both at the root and outside it.
5. Run the existing container distribution, ownership, source-relocation and
   affected workflow/documentation checks; inspect the exact source diff.
6. Qualify the resulting exact PR head in its own applicable CI. Preserve the
   predecessor's workload, schemas, scientific inputs and resource admission.

The original preview used Docker Compose v5.3.1 and all 30 complete configuration
pairs were equal. It invoked only `config`, without a Docker engine or network
workload. That preview is not an image build, GPU execution, new-head CI pass
or a claim that the remaining #1084-dependent families are complete.
