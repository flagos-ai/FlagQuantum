# Code of conduct

This project is a technical one with a small contributor base. Almost all of its
interaction happens in review: 271 merged pull requests, most of them authored and merged
by one account. The conduct that matters most here is therefore the conduct of review,
and this document is written for that.

The standard is the one already stated in `CONTRIBUTING.md`:

> Be constructive, not rude.
> Be open to feedback, not defensive.

This document promotes that standard into the file GitHub reads, and says what it means
in practice. It does not replace the standard with a longer one.

## Scope

This applies to issues, pull requests, review comments, commit messages, discussions,
and any other space operated by this project. It also applies when you represent the
project elsewhere.

## Expected

- **Criticize the change, not the person.** "This guard fails open because the check runs
  after the early return" is a review comment. "Did you even test this" is not.
- **Say what would make it acceptable.** A rejection without a stated condition is not a
  review.
- **Accept a correction about your own work.** Measured claims in this repository are
  checked, and being wrong about a line number or a count is normal.
- **Ask before assuming bad faith.** A surprising implementation is usually an
  incomplete contract rather than a hidden motive.
- **Keep disagreement about the code in the pull request.** Moving it to a private
  channel or a different thread removes the record other reviewers need.

## Not acceptable

- Personal attacks, insults, or derogatory comments about a contributor.
- Harassment of any kind, public or private.
- Sexualized language or imagery, and unwelcome sexual attention.
- Publishing someone's private information, including an email address or physical
  address, without their explicit permission.
- Deliberately misrepresenting what a change, a test, or a measurement establishes. In a
  repository that gates on evidence, a false claim of verification is a conduct issue and
  not only a technical one.
- Sustained disruption: reopening a settled decision without new information, or
  relitigating an approval after it has been recorded.

## Enforcement

Report a concern privately to **flagquantum@gmail.com**. Reports are read by the
maintainer. The reporter's identity is kept private unless they agree otherwise.

Consequences, applied in proportion to the behaviour and its repetition:

1. **Correction.** A private message stating what was unacceptable and why. A public
   apology may be requested.
2. **Warning.** A warning that continued behaviour has consequences, with no interaction
   with the people involved for a stated period.
3. **Temporary block.** A time-limited block from all project spaces.
4. **Permanent block.** A permanent block, for a pattern of violations or for harassment
   of an individual.

A maintainer involved in the incident does not decide its outcome alone where a second
person is available to decide it.

## What this document does not do

It does not adopt the [Contributor Covenant](https://www.contributor-covenant.org).
That text is a reasonable default, but adopting it is a maintainer decision this file
does not take on its own, because it requires naming an enforcement contact and choosing
a version — 2.1 and 3.0 both exist — and it introduces a differently licensed document
into an Apache-2.0 repository. Recorded as an open question in
[the governance document](GOVERNANCE.md) instead of imported silently.

It also does not claim an independent moderation body, an appeals committee, or a
confidential reporting system. None of those exists. What exists is one maintainer, one
email address, and the process above.

This file is a protected integration path (`team-ownership.toml`).
