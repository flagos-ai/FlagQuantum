# Release-guide source mapping

G2 item Q-0052 maps the complete Qualtran v0.7.0 guide
`dev_tools/how-to-cut-a-release.md`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`7d4e96808d109f354ad3e613277b6bb6e0d863a3`, into the native
[release runbook](RELEASING.md). This is a procedural adaptation, not an imported
release implementation or authorization to publish PR #1084. The upstream file
is unchanged in the reference environment. The runbook is original prose mapped
to existing FlagQuantum policy and tools, not a verbatim copy of the guide.

| Complete source responsibility | FlagQuantum owner and disposition |
| --- | --- |
| Four-step overview and packaging-guide link | Native runbook retains version, tag, verified package and next-development roles; native CI/CD commands replace generic tutorial upload instructions. |
| Version PR, main merge, stable version source | `flagquantum/version.py`, dynamic metadata and `RELEASE_POLICY.md`; prereleases remain permitted by native policy. |
| Coordinate other merges and review draft notes | Pin and review an exact candidate SHA on `main`; coordinate maintainers without inventing a repository-wide merge lock. |
| GitHub release, tag convention and patch maintenance branch | Review tag target and notes before the existing release event triggers CD. Native policy requires a green revision of `main`; the upstream `vx.y` patch-branch exception and mandatory `v` spelling are not introduced as new native rules. |
| Fresh main checkout, clean preview and packaging | Isolated candidate checkout and read-only worktree inspection; no destructive clean instruction for shared work. Existing reproducibility, contents, manifests and clean-install owners remain authoritative. |
| Upload to production PyPI rather than tutorial test repository | Existing CD PyPI publisher verifies its downloaded complete matrix. No manual upload path or credentials are added. |
| Bump to next minor development version | Follow-up PR at the sole version source; native semantic-version policy chooses the sequence rather than adopting the upstream arithmetic formula. |
| Announcement email | Reviewed release notes and authorized project channels; no upstream mailing list or automatic message sender is imported. |

Acceptance is limited to the complete documentation responsibility. Existing
package-policy tests exercise single-version metadata, distribution quarantine,
native extension requirements, checksums and verify-before-publish ordering;
release documentation tests exercise tier and claim boundaries. No tests merely
mirror this prose. No package was built, installed, tagged or published for this
documentation batch. Primitive counts, complex amplitudes, measurement histories
and workspace cleanup do not apply to a maintainer procedure. This closes no
algorithm, compiler, installer or ReadTheDocs execution gap.
