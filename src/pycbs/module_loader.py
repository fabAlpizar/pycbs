import importlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SchemeModuleLoader:
    """
    Resolves a scheme name to its corresponding module and loads it.
    Each module MUST expose: compute(params: dict)
    """

    SCHEME_MODULES: Dict[str, str] = {
        # --- HF-only schemes ---
        "HF_E": "pycbs.HF_component",
        "FELLER": "pycbs.HF_component",
        "JENSEN": "pycbs.HF_component",
        "KLOPPER": "pycbs.HF_component",
        "TRUHLAR_HF": "pycbs.HF_component",

        # --- Correlation-only schemes ---
        "BAKOWIES": "pycbs.corr_component",
        "HALKIER_HELGAKER": "pycbs.corr_component",
        "HUH_LEE": "pycbs.corr_component",
        "MARTIN": "pycbs.corr_component",
        "OANC": "pycbs.corr_component",
        "TRUHLAR_CORR": "pycbs.corr_component",

        # --- Full CBS extrapolation schemes ---
        "USTE1": "pycbs.uste1",
        "USTE2": "pycbs.uste2",
        "USPE": "pycbs.uspe",
        "TENSORIAL": "pycbs.tensorial",
        "FREQUENCY": "pycbs.frequency",
    }

    _cache: Dict[str, Any] = {}

    @classmethod
    def load_scheme(cls, scheme: str) -> Optional[Any]:
        scheme = scheme.upper().strip()

        if scheme in cls._cache:
            return cls._cache[scheme]

        module_path = cls.SCHEME_MODULES.get(scheme)
        if not module_path:
            raise ValueError(
                f"Unknown scheme '{scheme}'. "
                f"Available schemes: {', '.join(sorted(cls.SCHEME_MODULES))}"
            )

        module = importlib.import_module(module_path)

        if not hasattr(module, "compute"):
            raise AttributeError(
                f"Module '{module_path}' does not expose compute(params: dict)"
            )

        cls._cache[scheme] = module
        return module

    @classmethod
    def list_available_schemes(cls) -> list:
        return sorted(cls.SCHEME_MODULES.keys())
