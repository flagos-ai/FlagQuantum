# version.py
"""Version information for the distributed quantum device package."""

__version__ = "0.2.0"

VERSION_INFO = {
    "major": 0,
    "minor": 2,
    "patch": 0,
    "release_level": "final",  # "development", "alpha", "beta", "rc", "final"
}


def get_version() -> str:
    """Return the current version as a string."""
    return __version__
