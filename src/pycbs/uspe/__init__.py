from .USPE import USPE_CBS_extrapolation


def compute(params: dict):
    return USPE_CBS_extrapolation(
        zeta_HF1=params["HF1"],
        zeta_HF2=params["HF2"],
        zeta_E=params["E"],
        method=params["method"],
        constant=params["constant"],
        basis1=params["basis1"],
        basis2=params.get("basis2"),
    )
