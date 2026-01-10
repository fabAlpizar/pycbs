# src/pycbs/uspe/__init__.py
from .USPE import USPE_CBS_extrapolation

def compute(params: dict):
    # accept either zeta_hf1 names or hf1 (fallback)
    zeta_hf1 = params.get("zeta_hf1", params.get("hf1"))
    zeta_hf2 = params.get("zeta_hf2", params.get("hf2"))
    zeta_e = params.get("zeta_e", params.get("e", params.get("etot")))
    method = params.get("method")
    constant = params.get("constant", "normal")
    basis1 = params.get("basis1", params.get("basis"))

    if None in (zeta_hf1, zeta_hf2, zeta_e, method, basis1):
        raise ValueError("USPE compute() missing required parameters")

    zhf, zcor, ztot = USPE_CBS_extrapolation(zeta_hf1, zeta_hf2, zeta_e, method, constant, basis1, basis1)
    return {"EHF": zhf, "E_corr": zcor, "E_CBS": ztot}
