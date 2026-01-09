from .USTE2 import dictionaries, correlation_energy
from .USTE2 import hartree_fock_energy, dynamic_correlation_energy, CBS_extrapolation


def compute(params: dict):
    hf1 = params["HF1"]
    hf2 = params["HF2"]
    e1 = params["E1"]
    e2 = params["E2"]
    method = params["method"]

    b1, b2 = params["basis1"], params["basis2"]
    b3, b4 = params["basis3"], params["basis4"]

    _, dic = dictionaries(method, b1, b2, b3, b4)
    ecr1, ecr2 = correlation_energy(hf1, hf2, e1, e2)

    return CBS_extrapolation(
        hf1, hf2, ecr1, ecr2, dic, b1, b2, b3, b4
    )
