from .frequency import CBS_extrapolation, dictionaries


def compute(params: dict):
    hf1 = params["HF1"]
    hf2 = params["HF2"]
    f1 = params["F1"]
    f2 = params["F2"]
    method = params["method"]
    basis1 = params["basis1"]
    basis2 = params["basis2"]

    _, dic = dictionaries(method, basis1, basis2)

    return CBS_extrapolation(
        hf1, hf2, f1, f2, dic, basis1, basis2
    )
