# tensorial_properties1.py  (fixed & cleaned)

import math
from src.pycbs.basis import hf, dc1, dc2, dc3

# Function to retrieve dictionaries based on the selected method and two chosen databases
def dictionaries(metodo: str, basis1: str, basis2: str):
    """
    Return (hf_dict, correlation_dict) for the requested method and basis pairs.
    Raises ValueError when method or bases are unrecognized.
    """
    if metodo == "MP2":
        dic_correlacion = dc1
    elif metodo == "CCSD(T)":
        dic_correlacion = dc2
    elif metodo == "MP2+CCSD(T)":
        dic_correlacion = dc3
    else:
        raise ValueError("Invalid method: " + str(metodo))

    # allow basis2 == None -> use basis1
    if basis2 is None:
        basis2 = basis1

    for b in (basis1, basis2):
        if b not in hf:
            raise ValueError(f"The basis set '{b}' is not recognized in hf.")
        if b not in dic_correlacion:
            raise ValueError(f"The basis set '{b}' is not recognized in correlation database.")

    return hf, dic_correlacion


def select_constant(constant: str):
    """Return the constant table for either 'normal' or 'augmented' types."""
    a1 = {"MP2": 0.0111, "CCSD": 0.0073, "CCSD(T)": 0.0078}
    a2 = {"MP2": 0.0094, "CCSD": 0.0061, "CCSD(T)": 0.0065}

    if constant == "normal":
        return a1
    elif constant == "augmented":
        return a2
    else:
        raise ValueError("Invalid constant: " + str(constant))


def hartree_fock_energy(zeta_HF1: float, zeta_HF2: float, basis1: str, basis2: str) -> float:
    """
    HF extrapolation using the exponential ansatz provided in the original code.
    Returns the HF component (zeta_HF).
    """
    return zeta_HF2 + (
        (math.exp(2.284 * hf[basis1]))
        / (math.exp(2.284 * hf[basis2]) - math.exp(2.284 * hf[basis1]))
    ) * (zeta_HF2 - zeta_HF1)


# -------------------------
# USTE (two-point) helpers
# -------------------------
def dynamic_correlation_energy(zeta_cor1: float, zeta_cor2: float, dic_correlacion: dict, basis1: str, basis2: str) -> float:
    """
    Two-point inverse-cubic dynamic-correlation extrapolation:
      zeta_cor = zeta_cor2 + (b2^-3 / (b1^-3 - b2^-3)) * (zeta_cor2 - zeta_cor1)
    where b1/b2 are the hierarchical exponents stored in dic_correlacion for each basis.
    """
    b1 = float(dic_correlacion[basis1])
    b2 = float(dic_correlacion[basis2])
    denom = (b1 ** -3) - (b2 ** -3)
    if denom == 0:
        raise ZeroDivisionError("Denominator zero in dynamic correlation extrapolation.")
    return zeta_cor2 + ((b2 ** -3) / denom) * (zeta_cor2 - zeta_cor1)


def USTE_CBS_extrapolation(zeta_HF1: float, zeta_HF2: float, zeta_cor1: float, zeta_cor2: float,
                           dic_correlacion: dict, basis1: str, basis2: str):
    """
    Two-point CBS extrapolation using separate HF and correlation extrapolations.
    Returns (zeta_HF, zeta_cor, zeta_total).
    """
    zeta_HF = hartree_fock_energy(zeta_HF1, zeta_HF2, basis1, basis2)
    zeta_cor = dynamic_correlation_energy(zeta_cor1, zeta_cor2, dic_correlacion, basis1, basis2)
    return zeta_HF, zeta_cor, (zeta_HF + zeta_cor)


# -------------------------
# USPE (single-point) helpers
# -------------------------
def USPE_correlation_energy(method: str, basis1: str, constant: str, zeta_E: float) -> float:
    """
    Single-point (USPE) estimate of the correlation contribution:
      Ecr ~ (a_method * zeta_E) / (hierarchical_exponent(basis1)^3)
    """
    # unpack dictionaries properly
    _, dic_correlation = dictionaries(method, basis1, basis1)
    a_values = select_constant(constant)
    a = a_values.get(method)
    if a is None:
        raise KeyError(f"Method '{method}' not found in constants for USPE.")
    return (a * float(zeta_E)) / (float(dic_correlation[basis1]) ** 3)


def USPE_CBS_extrapolation(zeta_HF1: float, zeta_HF2: float, zeta_E: float,
                           method: str, constant: str, basis1: str, basis2: str):
    """
    Single-point CBS extrapolation for USPE-style flow.
    Returns (zeta_HF, zeta_cor, zeta_total) to keep consistent with USTE.
    """
    _, dic_correlation = dictionaries(method, basis1, basis2)
    a_values = select_constant(constant)
    a = a_values.get(method)
    if a is None:
        raise KeyError(f"Method '{method}' not found in constants for USPE.")
    zeta_HF = hartree_fock_energy(zeta_HF1, zeta_HF2, basis1, basis2)
    zeta_cor = (a * float(zeta_E)) / (float(dic_correlation[basis1]) ** 3)
    return zeta_HF, zeta_cor, (zeta_HF + zeta_cor)
