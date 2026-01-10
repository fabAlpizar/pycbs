"""
USPE wrapper module exposing compute(params: dict) -> dict

Canonical USPE (single-point only):
Required input:
  - scheme = USPE
  - method
  - basis
  - HF
  - Etot
Optional:
  - constant (default: "normal")
"""

from typing import Dict, Any
from .USPE import USPE_single_point

def compute(params: Dict[str, Any]) -> Dict[str, float]:
    # normalize keys (case-insensitive)
    sec = {k.lower(): v for k, v in params.items() if v is not None}

    # required parameters
    method = sec.get("method")
    basis = sec.get("basis")
    hf = sec.get("hf")
    etot = sec.get("etot")

    if method is None:
        raise ValueError("USPE missing required parameter: method")
    if basis is None:
        raise ValueError("USPE missing required parameter: basis")
    if hf is None:
        raise ValueError("USPE missing required parameter: HF")
    if etot is None:
        raise ValueError("USPE missing required parameter: Etot")

    constant = sec.get("constant", "normal")

    EHF, Ecorr, Etot_cbs = USPE_single_point(
        hf=float(hf),
        etot=float(etot),
        method=str(method).upper(),
        basis=str(basis),
        constant=str(constant)
    )

    return {
        "EHF": EHF,
        "E_corr": Ecorr,
        "E_CBS": Etot_cbs
    }
