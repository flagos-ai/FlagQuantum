"""Run all three source-size native adaptive Grover searches."""

from .workflow import run_workflow


def main() -> None:
    """Print finite-shot decisions, resources and original result display limits."""
    for search, limit in zip(run_workflow(), (3, 5, 32), strict=True):
        print(search.problem, "succeeded:", search.succeeded)
        for attempt in search.attempts:
            print(
                "Loop / repetitions / seed:",
                attempt.loop_index,
                attempt.repetitions,
                attempt.seed,
            )
            print(attempt.resources)
        for row in search.attempts[-1].outcomes[:limit]:
            print(row)


if __name__ == "__main__":
    main()
