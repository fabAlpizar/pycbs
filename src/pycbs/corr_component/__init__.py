# src/pycbs/corr_component/__init__.py
import math as mt
import inspect
from typing import Any, Dict

# Correlation schemes implemented with lowercase parameter names
def bakoules(ec_x: float, ec_y: float, x: int = 2, y: int = 3, beta: float = 3.877):
    num = ec_y * ((x + 1) ** -beta) - ec_x * ((y + 1) ** -beta)
    den = ((x + 1) ** -beta) - ((y + 1) ** -beta)
    return num / den


def halkier(ec_x: float, ec_y: float, x: int = 2, y: int = 3):
    num = ec_x * x**3 - ec_y * y**3
    den = x**3 - y**3
    return num / den


def huh_lee(ec_x: float, ec_y: float, x: int = 2, y: int = 3, beta: float = 0.220):
    num = ec_y * ((x + beta) ** -3) - ec_x * ((y + beta) ** -3)
    den = ((x + beta) ** -3) - ((y + beta) ** -3)
    return num / den


def martin(ec_x: float, ec_y: float, x: int = 2, y: int = 3, beta: float = 3.315):
    num = ec_y * ((x + 0.5) ** -beta) - ec_x * ((y + 0.5) ** -beta)
    den = ((x + 0.5) ** -beta) - ((y + 0.5) ** -beta)
    return num / den


def oan(ec_x: float, ec_y: float, beta: float = 2.086):
    num = ec_y * 27 - (beta**3) * ec_x
    den = 27 - (beta**3)
    return num / den


def truhlar_corr(ec_x: float, ec_y: float, x: int = 2, y: int = 3, beta: float = 2.751):
    num = ec_y * x**-beta - ec_x * y**-beta
    den = x**-beta - y**-beta
    return num / den


# Registry
CORR_SCHEMES = {
    "BAKOULES": bakoules,
    "HALKIER_HELGAKER": halkier,
    "HUH_LEE": huh_lee,
    "MARTIN": martin,
    "OAN": oan,
    "TRUHLAR_CORR": truhlar_corr,
}


def compute(params: Dict[str, Any]):
    scheme = params.get("scheme", "").upper()
    func = CORR_SCHEMES.get(scheme)
    if func is None:
        raise ValueError(f"Unknown correlation scheme: {scheme}")

    # Use inspect to only pass accepted args (functions use lowercase names)
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    kwargs = {k: v for k, v in params.items() if k in accepted}

    return func(**kwargs)
