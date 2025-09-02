"""
Version information for the Grassmann tensor package.
"""

__all__ = [
    "__version__",
]

try:
    from ._version import __version__
except ModuleNotFoundError:
    try:
        import importlib.metadata
        __version__ = importlib.metadata.version("parity")
    except importlib.metadata.PackageNotFoundError:
        __version__ = "0.0.0"
