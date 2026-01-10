# src/pycbs/tensorial/__init__.py
"""
Tensorial scheme entry-point.
Accepts a normalized params dict (lowercase keys; scheme/method uppercased).
This wrapper is tolerant with alternate input names (zeta_cor*, zeta_e*, e*, etc).
Returns a dict: {'EHF':..., 'E_corr':..., 'E_CBS':...}
"""

from .tensorial_properties1 import dictionaries, USTE_CBS_extrapolation, USPE_CBS_extrapolation

def _get_any(params, *names, default=None):
    """Return first value found in params among provided names."""
    for n in names:
        if n in params and params[n] is not None:
            return params[n]
    return default

def compute(params: dict):
    method = params.get("method")
    basis1 = params.get("basis1")
    basis2 = params.get("basis2", basis1)
    dc_scheme = params.get("dc_scheme", "USTE1")
    if isinstance(dc_scheme, str):
        dc_scheme = dc_scheme.upper()

    # USPE: single-point
    if dc_scheme == "USPE":
        # accept zeta names or hf1/hf2 and e or zeta_e
        zeta_hf1 = _get_any(params, "zeta_hf1", "hf1")
        zeta_hf2 = _get_any(params, "zeta_hf2", "hf2")
        zeta_e1 = _get_any(params, "zeta_e1", "zeta_cor1", "e1", "etot", "zeta_e")
        constant = params.get("constant", "normal")

        if None in (zeta_hf1, zeta_hf2, zeta_e1, method, basis1):
            raise ValueError("TENSORIAL (USPE) missing parameters: need zeta_hf1/zeta_hf2/zeta_e1, method and basis1")

        zhf, zcor, ztot = USPE_CBS_extrapolation(zeta_hf1, zeta_hf2, zeta_e1, method, constant, basis1, basis2)
        return {"EHF": float(zhf), "E_corr": float(zcor), "E_CBS": float(ztot)}

    # USTE1-style (two-point)
    zeta_hf1 = _get_any(params, "zeta_hf1", "hf1")
    zeta_hf2 = _get_any(params, "zeta_hf2", "hf2")
    zeta_e1 = _get_any(params, "zeta_e1", "zeta_cor1", "e1", "etot")
    zeta_e2 = _get_any(params, "zeta_e2", "zeta_cor2", "e2")

    if None in (zeta_hf1, zeta_hf2, zeta_e1, zeta_e2, method, basis1):
        raise ValueError("TENSORIAL (USTE1) missing parameters: need zeta_hf1, zeta_hf2, zeta_e1, zeta_e2, method and basis1")

    # get dictionaries and call the USTE helper which returns (EHF, zcor, total)
    hf_dict, corr_dict = dictionaries(method, basis1, basis2)
    EHF, dc, CBS = USTE_CBS_extrapolation(zeta_hf1, zeta_hf2, zeta_e1, zeta_e2, corr_dict, basis1, basis2)
    return {"EHF": float(EHF), "E_corr": float(dc), "E_CBS": float(CBS)}
