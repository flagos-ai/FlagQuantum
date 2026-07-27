# version.py
"""Version information for the distributed quantum device package."""

__version__ = "0.1.0"

VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "release_level": "development",  # "development", "alpha", "beta", "rc", "final"
}


def get_version() -> str:
    """Return the current version as a string."""
    return __version__
