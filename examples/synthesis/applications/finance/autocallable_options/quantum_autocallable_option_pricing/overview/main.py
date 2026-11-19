"""Run finite-grid autocallable pricing and the native circuit checks."""

from .workflow import print_report, run


def main() -> None:
    print_report(run())


if __name__ == "__main__":
    main()
