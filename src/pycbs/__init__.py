"""
pyCBS - Complete Basis Set Extrapolation Tool

Main package initialization
"""
import sys

__all__ = [
    'writer',
    'basis',
    'SchemeModuleLoader',
    'ConfigValidator',
    '__version__',
]

# Handle version
if sys.version_info[:2] >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version  # pragma:  no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma:  no cover

try:
    dist_name = "pycbs"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__author__ = "Alberto Guerra-Barroso, Fabio J. Delgado-Alpízar, Antonio J. C. Varandas"

# Import key modules (relative imports within package)
from .import writer
from .import basis
from .module_loader import SchemeModuleLoader
from .config_validator import ConfigValidator