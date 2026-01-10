# src/pycbs/uste2/__init__.py
from .USTE2 import dictionaries, correlation_energy, hartree_fock_energy, dynamic_correlation_energy, CBS_extrapolation

def compute(params: dict):
    method = params.get("method")
    basis1 = params.get("basis1")
    basis2 = params.get("basis2", basis1)
    basis3 = params.get("basis3", basis1)
    basis4 = params.get("basis4", basis2)

    hf1 = params.get("hf1")
    hf2 = params.get("hf2")
    e1 = params.get("e1")
    e2 = params.get("e2")

    if None in (method, basis1, basis2, basis3, basis4, hf1, hf2, e1, e2):
        raise ValueError("USTE2 compute() missing required parameters")

    hf_dict, corr_dict = dictionaries(method, basis1, basis2, basis3, basis4)

    ecr1, ecr2 = correlation_energy(hf1, hf2, e1, e2)
    EHF, dc, CBS = CBS_extrapolation(hf1, hf2, ecr1, ecr2, corr_dict, basis1, basis2, basis3, basis4)
    return {"EHF": EHF, "E_corr": dc, "E_CBS": CBS}
