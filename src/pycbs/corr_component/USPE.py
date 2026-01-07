# USPE.py  -- cleaned, consistent, and self-contained

import math
from typing import Tuple, Dict, Optional

from src.basis import hf, dc1, dc2, dc3

# ---------------------------------------------------------------------
# Dictionaries / constants
# ---------------------------------------------------------------------
def dictionaries(method: str, basis1: str, basis2: Optional[str] = None) -> Tuple[Dict, Dict]:
    """
    Return (hf_dict, correlation_dict) appropriate for the requested method.
    If basis2 is None it will be treated as basis1 for validation purposes.
    """
    if method == "MP2":
        dic_correlacion = dc1
    elif method == "CCSD(T)":
        dic_correlacion = dc2
    elif method == "MP2+CCSD(T)":
        dic_correlacion = dc3
    else:
        raise ValueError(f"Invalid method: {method!r}")

    if basis2 is None:
        basis2 = basis1

    # validate basis presence (helps produce early, clear errors)
    for b in (basis1, basis2):
        if b not in hf:
            raise ValueError(f"Basis '{b}' not found in hf lookup.")
        if b not in dic_correlacion:
            raise ValueError(f"Basis '{b}' not found in correlation lookup for method {method!r}.")

    return hf, dic_correlacion


def select_constant(constant: str) -> Dict[str, float]:
    """
    Return the table of constants for either 'normal' or 'augmented'.
    """
    a_normal = {"MP2": 0.0111, "CCSD": 0.0073, "CCSD(T)": 0.0078}
    a_aug = {"MP2": 0.0094, "CCSD": 0.0061, "CCSD(T)": 0.0065}

    if constant == "normal":
        return a_normal
    elif constant == "augmented":
        return a_aug
    else:
        raise ValueError(f"Invalid constant type: {constant!r} (expected 'normal' or 'augmented')")


# ---------------------------------------------------------------------
# HF extrapolation helper (keeps same formula you had)
# ---------------------------------------------------------------------
def hartree_fock_energy(zeta_HF1: float, zeta_HF2: float, basis1: str, basis2: str) -> float:
    """
    HF extrapolation using the exponential ansatz from the original code.
    Returns the extrapolated HF component (zeta_HF).
    """
    # Validate bases available in hf
    if basis1 not in hf or basis2 not in hf:
        raise ValueError(f"Basis '{basis1}' or '{basis2}' not found in hf lookup.")

    num = math.exp(2.284 * float(hf[basis1]))
    den = math.exp(2.284 * float(hf[basis2])) - math.exp(2.284 * float(hf[basis1]))
    if den == 0:
        raise ZeroDivisionError("Denominator zero in hartree_fock_energy computation.")
    return float(zeta_HF2) + (num / den) * (float(zeta_HF2) - float(zeta_HF1))


# ---------------------------------------------------------------------
# USPE single-point helpers
# ---------------------------------------------------------------------
def USPE_correlation_energy(method: str, basis1: str, constant: str, zeta_E: float) -> float:
    """
    Estimate the correlation contribution (single-point) using the USPE formula:
      zeta_cor = (a_method * zeta_E) / (hierarchical_exponent(basis1)^3)

    Returns a float.
    """
    _, dic_correlation = dictionaries(method, basis1, basis1)
    a_values = select_constant(constant)
    a = a_values.get(method)
    if a is None:
        raise KeyError(f"Method '{method}' not available in constant table.")
    exponent = float(dic_correlation[basis1])
    if exponent == 0:
        raise ZeroDivisionError("Hierarchical exponent is zero for basis " + basis1)
    return (float(a) * float(zeta_E)) / (exponent ** 3)


def USPE_CBS_extrapolation(zeta_HF1: float, zeta_HF2: float, zeta_E: float,
                           method: str, constant: str, basis1: str, basis2: Optional[str] = None):
    """
    Perform the USPE single-point CBS extrapolation and return a consistent tuple:
      (zeta_HF, zeta_cor, zeta_total)

    This keeps results consistent with USTE-style functions.
    """
    if basis2 is None:
        basis2 = basis1

    # get correlation dictionary (and validate)
    _, dic_correlation = dictionaries(method, basis1, basis2)
    a_values = select_constant(constant)
    a = a_values.get(method)
    if a is None:
        raise KeyError(f"Method '{method}' not available in constant table.")

    zeta_HF = hartree_fock_energy(zeta_HF1, zeta_HF2, basis1, basis2)
    exponent = float(dic_correlation[basis1])
    if exponent == 0:
        raise ZeroDivisionError("Hierarchical exponent is zero for basis " + basis1)
    zeta_cor = (float(a) * float(zeta_E)) / (exponent ** 3)
    zeta_total = zeta_HF + zeta_cor
    return float(zeta_HF), float(zeta_cor), float(zeta_total)
