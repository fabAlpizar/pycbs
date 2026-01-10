# src/pycbs/uspe/__init__.py
"""
USPE wrapper module exposing compute(params: dict) -> dict

This wrapper is tolerant of multiple input naming conventions and case:
 - keys are treated case-insensitively
 - accepts either:
    * single-point style: HF (single) + Etot
    * two-point style: hf1 & hf2 (or zeta_hf1 & zeta_hf2) + zeta_e (or e/etot)
 - method is required and will be uppercased before calling core routines.
 - constant is optional (defaults to 'normal').
"""

from typing import Any, Dict, Optional
from .USPE import USPE_CBS_extrapolation

def _to_float(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"Missing required numeric parameter: {name}")
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Parameter '{name}' must be numeric, got: {x!r}")

def compute(params: Dict[str, Any]) -> Dict[str, float]:
    """
    params: dictionary coming from CLI normalize_params (but wrapper handles case-insensitively).
    Returns: dict with keys "EHF", "E_corr", "E_CBS".
    """
    # Normalize incoming keys to lowercase for robust lookup
    sec = {str(k).strip().lower(): v for k, v in params.items() if v is not None}

    # Required: method and basis (validator already checks but double-check here)
    method = sec.get("method")
    if method is None:
        raise ValueError("USPE compute() missing required parameter: method")
    method_up = str(method).upper()

    # Basis: accept 'basis' or 'basis1'
    basis1 = sec.get("basis") or sec.get("basis1")
    if basis1 is None:
        raise ValueError("USPE compute() missing required parameter: basis / basis1")

    # Collect HF inputs:
    # - Accept two HF inputs if present: hf1/hf2 or zeta_hf1/zeta_hf2
    # - Otherwise accept single 'hf' or 'HF' (after normalization it's 'hf') and use it for both hf1/hf2
    hf1_raw = sec.get("zeta_hf1") or sec.get("hf1")
    hf2_raw = sec.get("zeta_hf2") or sec.get("hf2")
    hf_single_raw = sec.get("hf")  # single HF (USPE single-point style)

    if hf1_raw is not None and hf2_raw is not None:
        zhf1 = _to_float(hf1_raw, "hf1/zeta_hf1")
        zhf2 = _to_float(hf2_raw, "hf2/zeta_hf2")
    elif hf_single_raw is not None:
        zhf1 = zhf2 = _to_float(hf_single_raw, "hf")
    else:
        raise ValueError("USPE compute() missing HF input: provide either (hf1 & hf2) or single HF")

    # zeta_e: accept zeta_e, e, etot
    zeta_e_raw = sec.get("zeta_e") or sec.get("e") or sec.get("etot")
    if zeta_e_raw is None:
        raise ValueError("USPE compute() missing required correlation input: etot / e / zeta_e")
    zeta_e = _to_float(zeta_e_raw, "zeta_e/e/etot")

    # constant optional
    constant = sec.get("constant") or "normal"

    # Call core routine: USPE_CBS_extrapolation( zhf1, zhf2, zeta_e, method, constant, basis1, basis1 )
    zhf, zcor, ztot = USPE_CBS_extrapolation(zhf1, zhf2, zeta_e, method_up, constant, basis1, basis1)

    return {"EHF": float(zhf), "E_corr": float(zcor), "E_CBS": float(ztot)}
