# version.py
"""Version information for the distributed quantum device package."""

__version__ = "0.3.1"

VERSION_INFO = {
    "major": 0,
    "minor": 3,
    "patch": 1,
    "release_level": "final",  # "development", "alpha", "beta", "rc", "final"
}


def get_version() -> str:
    """Return the current version as a string."""
    return __version__
