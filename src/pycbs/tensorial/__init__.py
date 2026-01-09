from .tensorial_properties1 import USTE_CBS_extrapolation, dictionaries


def compute(params: dict):
    hf1 = params["HF1"]
    hf2 = params["HF2"]
    e1 = params["E1"]
    e2 = params["E2"]
    method = params["method"]
    basis1 = params["basis1"]
    basis2 = params.get("basis2")

    _, dic = dictionaries(method, basis1, basis2)

    return USTE_CBS_extrapolation(
        hf1, hf2, e1, e2, dic, basis1, basis2
    )
