"""
Module Loader:  Dynamically discovers and imports scheme modules
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SchemeModuleLoader:
    """
    Loads CBS extrapolation scheme modules dynamically.

    Scheme sources:
    - HF_component:  Feller, HF_E, Truhlar_hf, Klopper, Jensen
    - corr_component:  Bakoules, OAN, USTE1, USTE2, USPE, Truhlar_corr,
                      Martin, Halkier-Helgaker, Huh-Lee, tensorial_properties1
    - frequency: frequency
    """

    SCHEME_LOCATIONS = {
        # HF extrapolation schemes
        'FELLER': ('pycbs.HF_component. Feller', 'Feller_HF_extrapolation'),
        'HF_E': ('pycbs.HF_component.HF_E', 'hartree_fock_energy'),
        'TRUHLAR_HF': ('pycbs.HF_component.Truhlar_hf', 'Truhlar_HF_extrapolation'),
        'KLOPPER': ('pycbs.HF_component. Klopper', 'Klopper_HF_extrapolation'),
        'JENSEN': ('pycbs.HF_component.Jensen', 'Jensen_HF_extrapolation'),

        # Correlation extrapolation schemes
        'BAKOULES': ('pycbs. corr_component.Bakoules', 'Bakoules_HF_extrapolation'),
        'OAN': ('pycbs.corr_component. OAN', 'OAN_corr_extrapolation'),
        'USTE1': ('pycbs.corr_component.USTE1', None),  # Multi-function module
        'USTE2': ('pycbs.corr_component.USTE2', None),  # Multi-function module
        'USPE': ('pycbs.corr_component. USPE', None),  # Multi-function module
        'TRUHLAR_CORR': ('pycbs.corr_component.Truhlar_corr', 'Truhlar_corr_extrapolation'),
        'MARTIN': ('pycbs.corr_component.Martin', 'Martin_HF_extrapolation'),
        'HALKIER_HELGAKER': ('pycbs.corr_component. Halkier-Helgaker', 'Halkier_corr_extrapolation'),
        'HUH_LEE': ('pycbs.corr_component. Huh-Lee', 'Huh_Lee_HF_extrapolation'),
        'TENSORIAL': ('pycbs.corr_component.tensorial_properties1', None),  # Multi-function

        # Frequency schemes
        'FREQUENCY': ('pycbs.frequency.frequency', None),  # Multi-function module
    }

    _module_cache = {}

    @classmethod
    def load_module(cls, scheme_name: str) -> Optional[Any]:
        """
        Load a scheme module by name.

        Args:
            scheme_name:  Uppercase name of the scheme (e.g., 'USTE1', 'FELLER')

        Returns:
            Module object or None if not found
        """
        scheme_name = scheme_name.upper().strip()

        if scheme_name in cls._module_cache:
            return cls._module_cache[scheme_name]

        if scheme_name not in cls.SCHEME_LOCATIONS:
            logger.error(
                f"Scheme '{scheme_name}' not recognized.  Available:  {', '.join(cls.SCHEME_LOCATIONS.keys())}")
            return None

        module_path, _ = cls.SCHEME_LOCATIONS[scheme_name]

        try:
            module = importlib.import_module(module_path)
            cls._module_cache[scheme_name] = module
            logger.info(f"Loaded scheme module: {scheme_name} from {module_path}")
            return module
        except ImportError as e:
            logger.error(f"Failed to load scheme '{scheme_name}': {e}")
            return None

    @classmethod
    def get_function(cls, scheme_name: str, function_name: str) -> Optional[Callable]:
        """
        Get a specific function from a scheme module.

        Args:
            scheme_name: Name of the scheme
            function_name: Name of the function to retrieve

        Returns:
            Callable function or None
        """
        module = cls.load_module(scheme_name)
        if module is None:
            return None

        func = getattr(module, function_name, None)
        if not callable(func):
            logger.error(f"Function '{function_name}' not found or not callable in {scheme_name}")
            return None

        return func

    @classmethod
    def list_available_schemes(cls) -> list:
        """Return list of available scheme names."""
        return sorted(cls.SCHEME_LOCATIONS.keys())