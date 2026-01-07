"""
Module Loader:  Dynamically discovers and imports scheme modules.
Supports calling single-function modules (like Feller, OAN) and multi-function modules (USTE1, USPE).
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Dict
import inspect

logger = logging.getLogger(__name__)


class SchemeModuleLoader:
    """
    Loads CBS extrapolation scheme modules dynamically.

    Scheme sources:
    - HF_component: Feller, HF_E, Truhlar_hf, Klopper, Jensen
    - corr_component:  Bakoules, OAN, USTE1, USTE2, USPE, Truhlar_corr,
                      Martin, Halkier-Helgaker, Huh-Lee, tensorial_properties1
    - frequency:  frequency
    """

    # Map scheme names to module paths
    SCHEME_LOCATIONS = {
        # ===== HF EXTRAPOLATION SCHEMES (single-point functions) =====
        'FELLER': ('pycbs.HF_component. Feller', 'Feller_HF_extrapolation'),
        'HF_E': ('pycbs.HF_component.HF_E', 'hartree_fock_energy'),
        'TRUHLAR_HF': ('pycbs.HF_component.Truhlar_hf', 'Truhlar_HF_extrapolation'),
        'KLOPPER': ('pycbs.HF_component. Klopper', 'Klopper_HF_extrapolation'),
        'JENSEN': ('pycbs.HF_component.Jensen', 'Jensen_HF_extrapolation'),

        # ===== CORRELATION EXTRAPOLATION SCHEMES =====
        'BAKOULES': ('pycbs.corr_component.Bakoules', 'Bakoules_HF_extrapolation'),
        'OAN': ('pycbs.corr_component.OAN', 'OAN_corr_extrapolation'),
        'TRUHLAR_CORR': ('pycbs.corr_component.Truhlar_corr', 'Truhlar_corr_extrapolation'),
        'MARTIN': ('pycbs.corr_component.Martin', 'Martin_HF_extrapolation'),
        'HALKIER_HELGAKER': ('pycbs. corr_component.Halkier-Helgaker', 'Halkier_corr_extrapolation'),
        'HUH_LEE':  ('pycbs.corr_component.Huh-Lee', 'Huh_Lee_HF_extrapolation'),

        # ===== MULTI-FUNCTION MODULES (require special handling) =====
        'USTE1': ('pycbs.corr_component. USTE1', None),  # Multi-function module
        'USTE2': ('pycbs. corr_component.USTE2', None),  # Multi-function module
        'USPE': ('pycbs.corr_component.USPE', None),    # Multi-function module
        'TENSORIAL':  ('pycbs.corr_component.tensorial_properties1', None),  # Multi-function

        # ===== FREQUENCY SCHEMES =====
        'FREQUENCY': ('pycbs.frequency.frequency', None),  # Multi-function module
    }

    _module_cache: Dict[str, Any] = {}

    # Default parameters for optional arguments
    DEFAULT_PARAMS = {
        'alfa': 1.353,      # Feller default
        'beta': 2.086,      # OAN default
    }

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

        if scheme_name not in cls. SCHEME_LOCATIONS:
            logger.error(
                f"Scheme '{scheme_name}' not recognized. Available:  "
                f"{', '.join(cls.SCHEME_LOCATIONS. keys())}"
            )
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
            logger.error(
                f"Function '{function_name}' not found or not callable in {scheme_name}"
            )
            return None

        return func

    @classmethod
    def call_extrapolation_function(
        cls,
        scheme_name: str,
        Ehf_X: float,
        Ehf_Y: float,
        Ec_X:  Optional[float] = None,
        Ec_Y: Optional[float] = None,
        X:  Optional[float] = None,
        Y: Optional[float] = None,
        alfa: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> float:
        """
        Call an extrapolation function with automatic parameter mapping.

        This method intelligently detects which parameters the function needs
        and calls it with the appropriate arguments.

        Args:
            scheme_name: Name of the scheme (e.g., 'FELLER', 'OAN')
            Ehf_X, Ehf_Y: HF energies for basis sets X and Y
            Ec_X, Ec_Y: Correlation energies for basis sets X and Y
            X, Y: Basis set hierarchy numbers/exponents
            alfa: Optional parameter for exponential schemes (default: scheme-specific)
            beta: Optional parameter for polynomial schemes (default: scheme-specific)

        Returns:
            Extrapolated energy value

        Raises:
            ValueError: If required parameters are missing
            TypeError: If function cannot be called with provided parameters
        """
        scheme_name = scheme_name.upper().strip()
        module = cls.load_module(scheme_name)

        if module is None:
            raise ImportError(f"Could not load module for scheme:  {scheme_name}")

        # Find the function in the module
        func = cls._find_extrapolation_function(module, scheme_name)
        if func is None:
            raise AttributeError(f"No extrapolation function found in {scheme_name}")

        # Get function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Build argument dictionary based on what the function expects
        kwargs = {}
        arg_list = []

        for param_name in params:
            param_lower = param_name.lower()

            # Map parameter names to provided values
            if 'ehf' in param_lower and 'x' in param_lower and 'ehf_x' in [p.lower() for p in params]:
                kwargs[param_name] = Ehf_X
            elif 'ehf' in param_lower and 'y' in param_lower and 'ehf_y' in [p.lower() for p in params]:
                kwargs[param_name] = Ehf_Y
            elif 'ec' in param_lower and 'x' in param_lower and Ec_X is not None:
                kwargs[param_name] = Ec_X
            elif 'ec' in param_lower and 'y' in param_lower and Ec_Y is not None:
                kwargs[param_name] = Ec_Y
            elif param_lower == 'x' and X is not None:
                kwargs[param_name] = X
            elif param_lower == 'y' and Y is not None:
                kwargs[param_name] = Y
            elif param_lower == 'alfa' or param_lower == 'alpha':
                # Use provided alfa, or get default for this scheme
                default_alfa = cls._get_default_alfa(scheme_name)
                kwargs[param_name] = alfa if alfa is not None else default_alfa
            elif param_lower == 'beta':
                # Use provided beta, or get default for this scheme
                default_beta = cls._get_default_beta(scheme_name)
                kwargs[param_name] = beta if beta is not None else default_beta

        # Try calling with kwargs
        try:
            result = func(**kwargs)
            logger.debug(f"Called {scheme_name} with kwargs: {kwargs}")
            return float(result)
        except TypeError as e:
            logger.warning(f"kwargs call failed for {scheme_name}, trying positional args:  {e}")

        # Fallback:  try positional arguments in expected order
        positional_args = []
        for param_name in params:
            param_lower = param_name.lower()
            if 'ehf_x' in param_lower or ('ehf' in param_lower and 'x' in param_lower and Ehf_X is not None):
                positional_args.append(Ehf_X)
            elif 'ehf_y' in param_lower or ('ehf' in param_lower and 'y' in param_lower and Ehf_Y is not None):
                positional_args. append(Ehf_Y)
            elif 'ec_x' in param_lower or ('ec' in param_lower and 'x' in param_lower and Ec_X is not None):
                positional_args.append(Ec_X)
            elif 'ec_y' in param_lower or ('ec' in param_lower and 'y' in param_lower and Ec_Y is not None):
                positional_args.append(Ec_Y)
            elif param_lower == 'x' and X is not None:
                positional_args.append(X)
            elif param_lower == 'y' and Y is not None:
                positional_args. append(Y)
            elif param_lower == 'alfa' or param_lower == 'alpha':
                default_alfa = cls._get_default_alfa(scheme_name)
                positional_args.append(alfa if alfa is not None else default_alfa)
            elif param_lower == 'beta':
                default_beta = cls._get_default_beta(scheme_name)
                positional_args.append(beta if beta is not None else default_beta)

        try:
            result = func(*positional_args)
            logger. debug(f"Called {scheme_name} with positional args: {positional_args}")
            return float(result)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unable to call {scheme_name} with provided parameters: {e}")

    @staticmethod
    def _find_extrapolation_function(module:  Any, scheme_name: str) -> Optional[Callable]:
        """Find the main extrapolation function in a module"""
        # Common function name patterns
        patterns = [
            f"{scheme_name. lower()}_extrapolation",
            f"{scheme_name.lower()}_hf_extrapolation",
            f"{scheme_name.lower()}_corr_extrapolation",
            'extrapolation',
            'run',
        ]

        for pattern in patterns:
            for attr_name in dir(module):
                if attr_name.lower() == pattern and callable(getattr(module, attr_name)):
                    return getattr(module, attr_name)

        return None

    @staticmethod
    def _get_default_alfa(scheme_name: str) -> float:
        """Get default alfa parameter for exponential schemes"""
        defaults = {
            'FELLER': 1.353,
            'KLOPPER': 4.257,
            'JENSEN': 5.163,
        }
        return defaults.get(scheme_name, 1.353)

    @staticmethod
    def _get_default_beta(scheme_name: str) -> float:
        """Get default beta parameter for polynomial schemes"""
        defaults = {
            'OAN': 2.086,
            'TRUHLAR_CORR': 2.751,
            'BAKOULES': 3.877,
            'MARTIN': 3.315,
            'HUH_LEE': 0.220,
        }
        return defaults.get(scheme_name, 2.086)

    @classmethod
    def list_available_schemes(cls) -> list:
        """Return list of available scheme names."""
        return sorted(cls. SCHEME_LOCATIONS.keys())