"""Retain historical Python identifiers for relocated QEC definitions."""

from types import FunctionType


def _retain_module(module: str, *definitions: object) -> None:
    """Keep old pickle and caller-owned safe-global paths without wrappers."""
    for definition in definitions:
        if isinstance(definition, FunctionType):
            definition.__module__ = module
        elif isinstance(definition, type):
            source_module = definition.__module__
            definition.__module__ = module
            for member in vars(definition).values():
                if isinstance(member, property):
                    member = member.fget
                elif isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if (
                    isinstance(member, FunctionType)
                    and member.__module__ == source_module
                ):
                    member.__module__ = module
