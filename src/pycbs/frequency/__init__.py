# src/pycbs/frequency/__init__.py
from .frequency import dictionaries, CBS_extrapolation

def compute(params: dict):
    method = params.get("method")
    basis1 = params.get("basis1")
    basis2 = params.get("basis2", basis1)
    hf1 = params.get("hf1")
    hf2 = params.get("hf2")
    f1 = params.get("f1", params.get("F1", None))
    f2 = params.get("f2", params.get("F2", None))

    if None in (method, basis1, basis2, hf1, hf2, f1, f2):
        raise ValueError("FREQUENCY compute() missing parameters")

    hf_dict, corr_dict = dictionaries(method, basis1, basis2)
    EHF, dc, CBS = CBS_extrapolation(hf1, hf2, f1, f2, corr_dict, basis1, basis2)
    return {"EHF": EHF, "E_corr": dc, "E_CBS": CBS}
