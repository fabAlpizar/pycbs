from .USTE1 import dictionaries, correlation_energy, hartree_fock_energy
from .USTE1 import dynamic_correlation_energy, CBS_extrapolation


def compute(params: dict):
    hf1 = params["HF1"]
    hf2 = params["HF2"]
    e1 = params["E1"]
    e2 = params["E2"]
    method = params["method"]
    basis1 = params["basis1"]
    basis2 = params["basis2"]

    _, dic = dictionaries(method, basis1, basis2)
    ecr1, ecr2 = correlation_energy(hf1, hf2, e1, e2)

    return CBS_extrapolation(
        hf1, hf2, ecr1, ecr2, dic, basis1, basis2
    )
