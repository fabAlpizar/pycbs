# src/pycbs/tensorial/__init__.py
from .tensorial_properties1 import dictionaries, USTE_CBS_extrapolation, USPE_CBS_extrapolation

def compute(params: dict):
    method = params.get("method")
    basis1 = params.get("basis1")
    basis2 = params.get("basis2", basis1)
    dc_scheme = params.get("dc_scheme", "USTE1").upper()

    # If using USPE style single-point
    if dc_scheme == "USPE":
        zeta_hf1 = params.get("zeta_hf1")
        zeta_e1 = params.get("zeta_e1")
        constant = params.get("constant", "normal")
        if None in (zeta_hf1, zeta_e1, method, basis1):
            raise ValueError("TENSORIAL (USPE) missing parameters")
        result = USPE_CBS_extrapolation(zeta_hf1, zeta_hf1, zeta_e1, method, constant, basis1, basis2)
        # USPE returns (zeta_HF, zeta_cor, zeta_total)
        return {"EHF": result[0], "E_corr": result[1], "E_CBS": result[2]}

    # USTE1 style two-point tensorial
    zeta_hf1 = params.get("zeta_hf1")
    zeta_hf2 = params.get("zeta_hf2")
    zeta_e1 = params.get("zeta_e1")
    zeta_e2 = params.get("zeta_e2")
    if None in (zeta_hf1, zeta_hf2, zeta_e1, zeta_e2, method, basis1, basis2):
        raise ValueError("TENSORIAL (USTE1) missing parameters")

    hf_dict, corr_dict = dictionaries(method, basis1, basis2)
    EHF, dc, CBS = USTE_CBS_extrapolation(zeta_hf1, zeta_hf2, zeta_e1, zeta_e2, corr_dict, basis1, basis2)
    return {"EHF": EHF, "E_corr": dc, "E_CBS": CBS}
