# src/pycbs/HF_component/__init__.py
import math as mt
import inspect
from typing import Any, Dict, Callable

from pycbs.basis import hf

# -------------------------------------------------
# Individual implementations (use lowercase parameter names)
# -------------------------------------------------

def feller(ehf_x: float, ehf_y: float, x: int = 2, y: int = 3, alfa: float = 1.353):
    num = ehf_y * mt.exp(-alfa * x) - ehf_x * mt.exp(-alfa * y)
    den = mt.exp(-alfa * x) - mt.exp(-alfa * y)
    return num / den


def jensen(ehf_x: float, ehf_y: float, x: int = 2, y: int = 3, alfa: float = 5.163):
    num = ehf_y * (x + 1) * mt.exp(-alfa * mt.sqrt(x)) - ehf_x * (y + 1) * mt.exp(-alfa * mt.sqrt(y))
    den = (x + 1) * mt.exp(-alfa * mt.sqrt(x)) - (y + 1) * mt.exp(-alfa * mt.sqrt(y))
    return num / den


def klopper(ehf_x: float, ehf_y: float, x: int = 2, y: int = 3, alfa: float = 4.257):
    num = ehf_x * mt.exp(-alfa * mt.sqrt(y)) - mt.exp(-alfa * mt.sqrt(x)) * ehf_y
    den = mt.exp(-alfa * mt.sqrt(y)) - mt.exp(-alfa * mt.sqrt(x))
    return num / den


def truhlar(ehf_x: float, ehf_y: float, x: int = 2, y: int = 3, alfa: float = 3.337):
    num = ehf_y * x**-alfa - ehf_x * y**-alfa
    den = x**-alfa - y**-alfa
    return num / den


def hf_e(hf1: float, hf2: float, basis1: str, basis2: str):
    # use hf mapping from pycbs.basis
    return (
        hf1 * mt.exp(2.284 * hf[basis1])
        - hf2 * mt.exp(2.284 * hf[basis2])
    ) / (
        mt.exp(2.284 * hf[basis1])
        - mt.exp(2.284 * hf[basis2])
    )


# Registry mapping (scheme name -> function)
HF_SCHEMES = {
    "FELLER": feller,
    "JENSEN": jensen,
    "KLOPPER": klopper,
    "TRUHLAR_HF": truhlar,
    "HF_E": hf_e,
}


# Public entry point used by the loader
def compute(params: Dict[str, Any]):
    """
    params: normalized dict with lowercase keys (e.g. 'ehf_x', 'hf1', 'basis1', ...)
           and scheme value uppercased in params['scheme'] (e.g. 'FELLER').
    """
    scheme = params.get("scheme", "").upper()
    func = HF_SCHEMES.get(scheme)
    if func is None:
        raise ValueError(f"Unknown HF scheme: {scheme}")

    # Filter params to only the callable's accepted keyword names (function uses lowercase names)
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())

    # Build kwargs using lowercase keys already present in params
    kwargs = {k: v for k, v in params.items() if k in accepted}

    # Call and return
    return func(**kwargs)
