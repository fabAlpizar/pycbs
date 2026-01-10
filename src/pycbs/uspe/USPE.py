"""
USPE core routines (canonical single-point formulation)
"""

from typing import Dict, Tuple
from pycbs.basis import dc1, dc2, dc3

# ---------------------------------------------------------------------
# Correlation dictionaries
# ---------------------------------------------------------------------
def correlation_dictionary(method: str) -> Dict[str, float]:
    method = method.upper()
    if method == "MP2":
        return dc1
    if method == "CCSD(T)":
        return dc2
    if method == "MP2+CCSD(T)":
        return dc3
    raise ValueError(f"Invalid method for USPE: {method}")


def select_constant(constant: str) -> Dict[str, float]:
    a_normal = {"MP2": 0.0111, "CCSD": 0.0073, "CCSD(T)": 0.0078}
    a_aug = {"MP2": 0.0094, "CCSD": 0.0061, "CCSD(T)": 0.0065}

    c = constant.lower()
    if c == "normal":
        return a_normal
    if c == "augmented":
        return a_aug
    raise ValueError(f"Invalid constant: {constant}")


# ---------------------------------------------------------------------
# Canonical USPE single-point
# ---------------------------------------------------------------------
def USPE_single_point(
    hf: float,
    etot: float,
    method: str,
    basis: str,
    constant: str = "normal"
) -> Tuple[float, float, float]:
    """
    Canonical USPE:
      E_CBS = E_HF + (a_method * E_tot) / (hierarchical_exponent(basis)^3)
    """
    corr_dict = correlation_dictionary(method)
    if basis not in corr_dict:
        raise ValueError(f"Basis '{basis}' not available for method {method}")

    a_table = select_constant(constant)
    a = a_table.get(method)
    if a is None:
        raise ValueError(f"No USPE constant for method {method}")

    exponent = float(corr_dict[basis])
    if exponent == 0:
        raise ZeroDivisionError("Correlation exponent is zero")

    Ecorr = (a * float(etot)) / (exponent ** 3)
    Etot_cbs = float(hf) + Ecorr

    return float(hf), float(Ecorr), float(Etot_cbs)
