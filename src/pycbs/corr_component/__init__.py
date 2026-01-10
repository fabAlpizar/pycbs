import math as mt

# -------------------------------------------------
# Individual implementations
# -------------------------------------------------

def bakoules(Ec_X:float, Ec_Y:float, X=2, Y=3, beta=3.877):
    num = Ec_Y * ((X + 1)**-beta) - Ec_X * ((Y + 1)**-beta)
    den = ((X + 1)**-beta) - ((Y + 1)**-beta)
    return num / den


def halkier(Ec_X:float, Ec_Y:float, X=2, Y=3):
    num = Ec_X * X**3 - Ec_Y * Y**3
    den = X**3 - Y**3
    return num / den


def huh_lee(Ec_X:float, Ec_Y:float, X=2, Y=3, beta=0.220):
    num = Ec_Y * ((X + beta)**-3) - Ec_X * ((Y + beta)**-3)
    den = ((X + beta)**-3) - ((Y + beta)**-3)
    return num / den


def martin(Ec_X:float, Ec_Y:float, X=2, Y=3, beta=3.315):
    num = Ec_Y * ((X + 0.5)**-beta) - Ec_X * ((Y + 0.5)**-beta)
    den = ((X + 0.5)**-beta) - ((Y + 0.5)**-beta)
    return num / den


def oan(Ec_X:float, Ec_Y:float, beta=2.086):
    num = Ec_Y * 27 - (beta**3) * Ec_X
    den = 27 - (beta**3)
    return num / den


def truhlar(Ec_X:float, Ec_Y:float, X=2, Y=3, beta=2.751):
    num = Ec_Y * X**-beta - Ec_X * Y**-beta
    den = X**-beta - Y**-beta
    return num / den


# -------------------------------------------------
# Registry
# -------------------------------------------------

CORR_SCHEMES = {
    "BAKOULES": bakoules,
    "HALKIER_HELGAKER": halkier,
    "HUH_LEE": huh_lee,
    "MARTIN": martin,
    "OAN": oan,
    "TRUHLAR_CORR": truhlar,
}


# -------------------------------------------------
# Public entry point
# -------------------------------------------------

def compute(params: dict):
    scheme = params["scheme"].upper()
    func = CORR_SCHEMES.get(scheme)

    if func is None:
        raise ValueError(f"Unknown correlation scheme: {scheme}")

    kwargs = {
        k: v for k, v in params.items()
        if k not in {"scheme", "method"}
    }

    return func(**kwargs)
