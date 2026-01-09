import math as mt
from pycbs.basis import hf

# -------------------------------------------------
# Individual implementations
# -------------------------------------------------

def feller(Ehf_X, Ehf_Y, X=2, Y=3, alfa=1.353):
    num = Ehf_Y * mt.exp(-alfa * X) - Ehf_X * mt.exp(-alfa * Y)
    den = mt.exp(-alfa * X) - mt.exp(-alfa * Y)
    return num / den


def jensen(Ehf_X, Ehf_Y, X=2, Y=3, alfa=5.163):
    num = Ehf_Y * (X + 1) * mt.exp(-alfa * mt.sqrt(X)) \
        - Ehf_X * (Y + 1) * mt.exp(-alfa * mt.sqrt(Y))
    den = (X + 1) * mt.exp(-alfa * mt.sqrt(X)) \
        - (Y + 1) * mt.exp(-alfa * mt.sqrt(Y))
    return num / den


def klopper(Ehf_X, Ehf_Y, X=2, Y=3, alfa=4.257):
    num = Ehf_X * mt.exp(-alfa * mt.sqrt(Y)) \
        - mt.exp(-alfa * mt.sqrt(X)) * Ehf_Y
    den = mt.exp(-alfa * mt.sqrt(X)) - mt.exp(-alfa * mt.sqrt(Y))
    return num / den


def truhlar(Ehf_X, Ehf_Y, X=2, Y=3, alfa=3.337):
    num = Ehf_Y * X**-alfa - Ehf_X * Y**-alfa
    den = X**-alfa - Y**-alfa
    return num / den


def hf_e(HF1, HF2, basis1, basis2):
    return (
        HF1 * mt.exp(2.284 * hf[basis1])
        - HF2 * mt.exp(2.284 * hf[basis2])
    ) / (
        mt.exp(2.284 * hf[basis1])
        - mt.exp(2.284 * hf[basis2])
    )


# -------------------------------------------------
# Registry
# -------------------------------------------------

HF_SCHEMES = {
    "FELLER": feller,
    "JENSEN": jensen,
    "KLOPPER": klopper,
    "TRUHLAR_HF": truhlar,
    "HF_E": hf_e,
}


# -------------------------------------------------
# Public entry point
# -------------------------------------------------

def compute(params: dict):
    scheme = params["scheme"].upper()
    func = HF_SCHEMES.get(scheme)

    if func is None:
        raise ValueError(f"Unknown HF scheme: {scheme}")

    kwargs = {
        k: v for k, v in params.items()
        if k not in {"scheme", "method"}
    }

    return func(**kwargs)
