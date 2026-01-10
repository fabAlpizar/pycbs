# src/pycbs/uspe/__init__.py
"""
USPE wrapper module exposing compute(params: dict) -> dict

The CLI normalizes config keys to lowercase. This compute() accepts
lowercase keys and common alternative names and coerces numeric values.
Return: {"EHF": ..., "E_corr": ..., "E_CBS": ...}
"""

from .USPE import USPE_CBS_extrapolation
from typing import Any, Dict

# Helper: coerce to float (clear error if not numeric)
def _to_float(x: Any, name: str):
    if x is None:
        raise ValueError(f"Missing required numeric parameter: {name}")
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Parameter '{name}' must be numeric, got: {x!r}")

def compute(params: Dict) -> Dict:
    """
    params: dictionary with normalized keys (CLI uses lowercase keys).
    Accepts several synonyms:
      - hf values: 'zeta_hf1' or 'hf1'  /  'zeta_hf2' or 'hf2'
      - energy: 'zeta_e' or 'e' or 'etot'
      - basis: 'basis1' or 'basis'
      - method: 'method' (already uppercased by normalize_params in CLI)
      - constant: optional (defaults to 'normal')
    """

    # Accept lower-case keys (CLI normalize_params lowercases keys)
    # But also permit callers that pass uppercase keys
    get = lambda *names: next((params.get(n) for n in names if n in params and params.get(n) is not None), None)

    zeta_hf1_raw = get("zeta_hf1", "hf1", "HF1", "ZETA_HF1")
    zeta_hf2_raw = get("zeta_hf2", "hf2", "HF2", "ZETA_HF2")
    zeta_e_raw = get("zeta_e", "e", "etot", "E", "ETOT", "ZETA_E")
    method = get("method", "METHOD")
    constant = get("constant", "CONSTANT") or "normal"
    basis1 = get("basis1", "basis", "BASIS1", "BASIS")

    if method is None:
        raise ValueError("USPE compute() missing required parameter: method")
    # method asserted by validator earlier, but we double-check
    # normalize method to usual case used by dictionaries
    method_norm = method.upper()

    if basis1 is None:
        raise ValueError("USPE compute() missing required parameter: basis / basis1")

    # Coerce numerics
    zhf1 = _to_float(zeta_hf1_raw, "zeta_hf1/hf1")
    zhf2 = _to_float(zeta_hf2_raw, "zeta_hf2/hf2")
    ze = _to_float(zeta_e_raw, "zeta_e/e/etot")

    # call core
    zhf, zcor, ztot = USPE_CBS_extrapolation(zhf1, zhf2, ze, method_norm, constant, basis1, basis1)

    return {"EHF": zhf, "E_corr": zcor, "E_CBS": ztot}
