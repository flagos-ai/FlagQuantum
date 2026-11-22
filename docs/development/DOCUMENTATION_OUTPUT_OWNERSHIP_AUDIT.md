# Complete documentation output ignore policy

Q-0121 covers the complete `docs/.gitignore` from Qualtran v0.7.0 at revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, original Git blob
`a82e2038fc266b5b92c91c29f7b1de80089ddd2a`.

The source has four rules, covering Sphinx output, generated API contents and
all Markdown/notebook files below docs. Its build guide describes generated
content being committed on a branch with a different ignore policy. These
source rules are tested unchanged in a real isolated Git repository.

The native docs tree contains authored Markdown. Its `docs/.gitignore` maps the
same responsibilities to existing generated-output owners. It ignores only
Sphinx `_build`, the published component catalog, the generated API namespace
and its root page, and the generated API TOC. These paths are already owned by
the accepted site, notebook publisher and reference renderers.

The real source/native Git checks enumerate ignored and visible files. Native
API and contributor Markdown, concept guides, authored notebooks and the root
RST remain visible. All generated fixture paths are ignored. Explicit scoped
force-add in the isolated fixture demonstrates that maintainers can stage a
selected generated page for a publication checkout. The test does not contact
Git remotes or publish a release.

This is an explicit authored-versus-generated layout adaptation. It introduces
no new output registry or duplicate build owner. The complete source policy is
accounted for without hiding the native authored documentation tree. The
broader source branch and hosting workflow retains its independent deployment
scope in the documentation build guide.

Ignore rules emit no quantum primitives or model estimates and own no quantum
state, measurement branches or workspace. Existing algorithm-owner evidence
remains authoritative for those categories.

The combined documentation batch passes 67 concentrated and 29 integration
cases. Strict typing covers both new tests. Black, Ruff, Integration scope,
architecture, internal IR, protected API baseline, local links and whitespace
checks pass. Four existing dependency/fork warnings remain visible.
