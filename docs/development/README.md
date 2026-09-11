# Development

Policies and workflows for changing, testing, and releasing FlagQuantum.

- [Python engineering standard](PYTHON_ENGINEERING_STANDARD.md)
- [Testing manual](TESTING.md)
- [Code organization](CODE_ORGANIZATION.md)
- [Repository governance](REPOSITORY_GOVERNANCE.md)
- [Multi-team linked-worktree development](MULTI_TEAM_DEVELOPMENT.md)
- [Multi-team handoff template](TEAM_HANDOFF_TEMPLATE.md)
- [Public API protection](PUBLIC_API_PROTECTION.md)
- [Stable Core API change proposal](API_CHANGE_PROPOSAL_001_STABLE_CORE.md)
- [Dependency policy](DEPENDENCY_POLICY.md)
- [Version and release policy](RELEASE_POLICY.md)
- [Git history recovery](GIT_HISTORY_RECOVERY.md)

Read the repository-level [contribution guide](../../CONTRIBUTING.md) and
[agent operating manual](../../AGENTS.md) before making changes.

## Writing README files

Write for the reader's next decision, not a mandatory document template.

| Location | Reader's question | Include |
| --- | --- | --- |
| Repository root | What can I build, and where do I start? | Product purpose, a complete small example, and documentation links. |
| Module | What belongs here, and how do I change it? | Design intent, ownership, key entry points, and focused verification. |
| Examples and operations | How do I run this correctly? | Prerequisites, commands, expected results, and recovery or interpretation notes. |
| Evidence and decisions | What happened, and what does it establish? | Provenance, recorded facts, approval scope, and claim boundaries. |

Keep a module's shortest usage or change path in its README. Split out a guide
only when detailed protocol, migration, or experiment material would obscure
that path; do not create a second file for a short source map.

Describe unfinished capabilities as design goals and give them behavioral
acceptance criteria. Runnable examples must use existing interfaces; speculative
API sketches must be labeled non-executable. A design document neither changes
protected APIs nor establishes hardware support.

Link to the capability catalog for support levels. Preserve historical numbers
and approval records as facts. Prefer specific checks such as gradient agreement
or restart equivalence over repeated claims of rigor and completeness.
