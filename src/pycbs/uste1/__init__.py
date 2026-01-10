# src/pycbs/uste1/__init__.py
from .USTE1 import dictionaries, correlation_energy, hartree_fock_energy, dynamic_correlation_energy, CBS_extrapolation

def compute(params: dict):
    """
    Expect lowercase keys in params:
      scheme (uppercased by normalize), method, basis1, basis2, hf1, hf2, e1, e2
    """
    method = params.get("method")
    basis1 = params.get("basis1")
    basis2 = params.get("basis2", basis1)

    hf1 = params.get("hf1")
    hf2 = params.get("hf2")
    e1 = params.get("e1")
    e2 = params.get("e2")

    if None in (method, basis1, basis2, hf1, hf2, e1, e2):
        raise ValueError("USTE1 compute() missing required parameters")

    hf_dict, corr_dict = dictionaries(method, basis1, basis2)

    # correlation_energy, hartree_fock_energy and CBS_extrapolation expect positional args;
    # pass the numeric values as positional args
    ecr1, ecr2 = correlation_energy(hf1, hf2, e1, e2)
    EHF, dc, CBS = CBS_extrapolation(hf1, hf2, ecr1, ecr2, corr_dict, basis1, basis2)
    return {"EHF": EHF, "E_corr": dc, "E_CBS": CBS}
