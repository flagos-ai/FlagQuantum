# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/prepared_env.py.

"""Commit comparison context for internal development checks."""

from tools.development import process


class PreparedEnvironment:
    """Describe the checkout and revisions used by a development check.

    A missing actual revision compares against local uncommitted changes.
    A missing destination delegates working-directory selection to subprocess.
    """

    def __init__(
        self,
        actual_commit_id: str | None,
        compare_commit_id: str,
        destination_directory: str | None,
    ) -> None:
        self.actual_commit_id = actual_commit_id
        self.compare_commit_id = compare_commit_id
        if self.compare_commit_id == self.actual_commit_id:
            self.compare_commit_id += "~1"
        self.destination_directory = destination_directory

    def get_changed_files(self) -> list[str]:
        """Return changed tracked paths relative to the comparison checkout."""
        actual = [] if self.actual_commit_id is None else [self.actual_commit_id]
        output = process.output_of(
            ["git", "diff", "--name-only", self.compare_commit_id, *actual, "--"],
            cwd=self.destination_directory,
            timeout=15,
        )
        return [line for line in output.split("\n") if line.strip()]
