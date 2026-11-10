"""Compatibility entry point; use ``python -m examples.realtime.device_call``."""

from examples.realtime.device_call import main, majority_decode

__all__ = ("main", "majority_decode")

if __name__ == "__main__":
    main()
