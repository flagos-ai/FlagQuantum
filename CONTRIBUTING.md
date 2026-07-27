# Contributing to FlagQuantum

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to **FlagQuantum**, hosted under [FlagOS](https://github.com/flagos-ai) on GitHub. These are mostly guidelines, not rigid rules. Use your best judgment, and feel free to propose changes to this document via a pull request.

#### Table of Contents

- [Code of Conduct](#code-of-conduct)
- [What I should know before I get started?](#what-should-i-know-before-i-get-started)
  - [FlagQuantum Overview](#flagquantum-overview)
  - [Design Decisions](#design-decisions)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs & Suggesting Improvements](#reporting-bugs--suggesting-improvements)
  - [Pull Requests](#pull-requests)
- [Styleguides](#styleguides)
  - [Git Commit Messages](#git-commit-messages)
  - [Languages & Code Style](#languages--code-style)

---

## Code of Conduct

Be constructive, not rude.
Be open to feedback, not defensive.
That’s it. Let’s build something great together.

---

## What should I know before I get started?

### FlagQuantum Overview

FlagQuantum bridges quantum computing and differentiable machine learning in a **scalable**, **hardware-agnostic** way (to the extent that PyTorch supports).
It is designed for researchers, developers, and quantum enthusiasts who want to integrate quantum layers into modern ML pipelines.

### Design Decisions

We aim to keep FlagQuantum:

- **Hybrid** – benefiting from both Object-Oriented and Functional programming styles where appropriate
- **Extensible** – easy to add new quantum operators, simulators, or backends
- **Modular** – components can be used independently
- **Parsimonious** – no unnecessary dependencies or abstractions

Major design decisions may be documented in the `README.md`. If you’re unsure why something works a certain way, check there first.

---

## How Can I Contribute?

### Reporting Bugs & Suggesting Improvements

This section guides you through submitting a bug report or feature suggestion. Following these guidelines helps maintainers and the community understand your report, reproduce issues, and find related tickets.

> **Note:** If you find a **closed** issue that seems similar to what you're experiencing, open a new one and link to the original issue in the body.

#### Before Submitting an Issue

- **Search** the [issue tracker](https://github.com/flagos-ai/FlagQuantum/issues) to see if the problem or idea has already been reported.
- If it exists **and is still open**, add a comment or 👍 instead of opening a new issue.

#### How to Submit a Good Issue

Bugs and suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). When creating an issue, please include:

- **A clear, descriptive title**
- **Setup & reproduction steps** – OS, Python version, package manager, exact commands, etc.
- **Observed behavior** – what actually happens
- **Expected behavior** – what you thought should happen
- **Reproducibility** – does it happen every time? If not, describe frequency and conditions

---

### Pull Requests

To get your contribution reviewed and merged:

1. Follow the [styleguides](#styleguides)
2. Ensure your changes do not break existing functionality
3. If relevant, add or update tests
4. The maintainers may request additional design work, tests, or changes before accepting the PR

---

## Styleguides

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line ≤ 72 characters
- Reference issues or PRs after the first line
- Feel free to start with an emoji:

  | Emoji | Code | Meaning |
  |-------|------|---------|
  | 🎨 | `:art:` | code style / structure |
  | 🐎 | `:racehorse:` | performance |
  | 🚱 | `:non-potable_water:` | memory leak fix |
  | 📝 | `:memo:` | documentation |
  | 🐛 | `:bug:` | bug fix |
  | 🔥 | `:fire:` | remove code/files |
  | ✅ | `:white_check_mark:` | add or update tests |
  | ⬆️ / ⬇️ | `:arrow_up:` / `:arrow_down:` | upgrade / downgrade dependencies |

### Languages & Code Style

- Documentation: **Markdown**
- Python code: follow [PEP 8](https://peps.python.org/pep-0008/) (especially naming conventions)
- Use meaningful variable and function names
- Keep functions small and focused

---

**Thank you** for contributing to FlagQuantum! 🙌
