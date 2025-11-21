# tensorial_properties1.py
"""
Tensorial properties helper module for pyCBS.

This module provides:
- dictionaries(method, basis1, basis2=None) -> (hf_dict, corr_dict)
- hartree_fock_energy(zeta_HF1, zeta_HF2, basis1, basis2)
- dynamic_correlation_energy_uste(zeta_cor1, zeta_cor2, corr_dict, basis1, basis2)
- correlation_energy_uspe(zeta_HF, zeta_E) -> correlation contribution
- CBS_extrapolation(...) -> will detect call-style:
    * USTE style: CBS_extrapolation(zeta_HF1, zeta_HF2, zeta_cor1, zeta_cor2, corr_dict, basis1, basis2)
      returns (zeta_HF, zeta_cor, zeta_total)
    * USPE style: CBS_extrapolation(zeta_HF, zeta_E, method, constant, basis) -> returns single zeta_total
"""
from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Optional

# import dictionaries from your basis module (assumes basis.py in same package)
from basis import hf, dc1, dc2, dc3  # hf is dict, dc* are dicts keyed by basis names

# human-friendly unicode zeta label (not used computationally)
zeta_label = "ζ"


# ----------------------------
# Lookup / dictionaries
# ----------------------------
def dictionaries(method: str, basis1: str, basis2: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Return (hf_dict, corr_dict) for the given method and bases.

    If basis2 is None, this still returns hf and the correlation dictionary corresponding
    to `method` (useful for USPE-style single-basis use).
    """
    method = method.strip()
    if method == "MP2":
        dic_correlacion = dc1
    elif method == "CCSD(T)":
        dic_correlacion = dc2
    elif method == "MP2+CCSD(T)":
        dic_correlacion = dc3
    else:
        raise ValueError(f"[tensorial_properties1] Unsupported method '{method}'")

    # Basic validation: ensure hf and corr dicts exist for provided basis keys
    if basis1 not in hf:
        raise ValueError(f"[tensorial_properties1] Unrecognized basis '{basis1}' in hf dictionary")
    if basis1 not in dic_correlacion:
        raise ValueError(f"[tensorial_properties1] Unrecognized basis '{basis1}' in correlation dictionary")

    if basis2 is not None:
        if basis2 not in hf:
            raise ValueError(f"[tensorial_properties1] Unrecognized basis '{basis2}' in hf dictionary")
        if basis2 not in dic_correlacion:
            raise ValueError(f"[tensorial_properties1] Unrecognized basis '{basis2}' in correlation dictionary")

    return hf, dic_correlacion


# ----------------------------
# Constants selector (for USPE use)
# ----------------------------
def select_constant(constant: str) -> Dict[str, float]:
    """
    Return a dict of 'a' values for different methods depending on 'constant' type.
    Allowed constant values: "normal", "augmented".
    """
    a1 = {'MP2': 0.0111, 'CCSD': 0.0073, 'CCSD(T)': 0.0078}
    a2 = {'MP2': 0.0094, 'CCSD': 0.0061, 'CCSD(T)': 0.0065}

    constant = constant.strip().lower()
    if constant == "normal":
        return a1
    elif constant == "augmented":
        return a2
    else:
        raise ValueError(f"[tensorial_properties1] Invalid constant '{constant}' (use 'normal' or 'augmented')")


# ----------------------------
# Correlation / HF helpers
# ----------------------------
def run_tensorial_from_section(section: dict) -> tuple:
    """
    High-level helper that accepts an INI-like mapping (section) and returns
    (zeta_HF, zeta_cor, zeta_total) for USTE-style or (zeta_HF, zeta_cor, zeta_total)
    for USPE-style (zeta_cor may be computed/None).
    Expected keys:
      - method, basis1
      - For USTE: zeta_HF1, zeta_HF2, zeta_E1, zeta_E2
      - For USPE: zeta_HF1 (or zeta_HF), zeta_E1 (or zeta_E)
      - dc_scheme (optional): "USTE" or "USPE"
      - constant (for USPE): optional
    """
    method = section.get("method")
    basis1 = section.get("basis1")
    dc_scheme = section.get("dc_scheme", section.get("dc", "USTE1")).strip().upper()

    if dc_scheme.startswith("USPE"):
        raw_hf = section.get("zeta_HF1", section.get("zeta_HF", None))
        raw_e  = section.get("zeta_E1", section.get("zeta_E", None))
        if raw_hf is None or raw_e is None:
            raise KeyError("USPE-style tensorial requires 'zeta_HF1'/'zeta_HF' and 'zeta_E1'/'zeta_E'")
        zeta_HF = float(raw_hf)
        zeta_E  = float(raw_e)
        # try to compute via our internal USPE CBS_extrapolation
        const = section.get("constant", section.get("constant_type", "normal"))
        # use our dispatcher (it returns a float)
        zeta_total = CBS_extrapolation(zeta_HF, zeta_E, method, const, basis1)
        zeta_cor = correlation_energy_uspe(zeta_HF, zeta_E)
        return float(zeta_HF), float(zeta_cor), float(zeta_total)

    else:
        # USTE 2-point
        raw_hf1 = section.get("zeta_HF1", section.get("zeta_HF", None))
        raw_hf2 = section.get("zeta_HF2", section.get("zeta_HF_2", None))
        raw_e1  = section.get("zeta_E1",  section.get("zeta_E", None))
        raw_e2  = section.get("zeta_E2",  section.get("zeta_E_2", None))
        if raw_hf1 is None or raw_hf2 is None or raw_e1 is None or raw_e2 is None:
            raise KeyError("USTE-style tensorial requires zeta_HF1/zeta_HF2 and zeta_E1/zeta_E2 keys")
        zeta_HF1 = float(raw_hf1); zeta_HF2 = float(raw_hf2)
        zeta_E1 = float(raw_e1);   zeta_E2 = float(raw_e2)
        hf_dict, corr_dict = dictionaries(method, basis1, section.get("basis2", None))
        # get correlations per basis
        zcr1, zcr2 = correlation_energy_uste(zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
        zeta_HF, zeta_cor, zeta_total = CBS_extrapolation(zeta_HF1, zeta_HF2, zcr1, zcr2, corr_dict, basis1, section.get("basis2"))
        return zeta_HF, zeta_cor, zeta_total


def correlation_energy_uste(zeta_HF1: float, zeta_HF2: float, zeta_E1: float, zeta_E2: float) -> Tuple[float, float]:
    """
    USTE-style: take two HF and two total energies and return two correlation contributions:
    (zeta_cor1, zeta_cor2) = E - HF for each basis/level.
    """
    zeta_cor1 = float(zeta_E1) - float(zeta_HF1)
    zeta_cor2 = float(zeta_E2) - float(zeta_HF2)
    return zeta_cor1, zeta_cor2


def hartree_fock_energy(zeta_HF1: float, zeta_HF2: float, basis1: str, basis2: str) -> float:
    """
    HF extrapolation tailored for zeta values (keeps your original formula).
    """
    # protect/convert
    a = float(hf[basis1])
    b = float(hf[basis2])
    # avoid division by zero
    denom = math.exp(2.284 * b) - math.exp(2.284 * a)
    if denom == 0.0:
        raise ZeroDivisionError("[tensorial_properties1] Zero denominator in hartree_fock_energy")
    zeta_HF = float(zeta_HF2) + ((math.exp(2.284 * a) / denom) * (float(zeta_HF2) - float(zeta_HF1)))
    return zeta_HF


def dynamic_correlation_energy_uste(zeta_cor1: float, zeta_cor2: float, dic_correlacion: Dict[str, float],
                                    basis1: str, basis2: str) -> float:
    """
    USTE inverse-cubic dynamic-correlation extrapolation for zeta:
    zeta_cor = zeta_cor2 + (b2^-3 / (b1^-3 - b2^-3)) * (zeta_cor2 - zeta_cor1)
    where dic_correlacion[basis] provides an exponent-like numeric value.
    """
    a1 = float(dic_correlacion[basis1])
    a2 = float(dic_correlacion[basis2])
    denom = (a1 ** -3) - (a2 ** -3)
    if denom == 0.0:
        raise ZeroDivisionError("[tensorial_properties1] Zero denominator in dynamic_correlation_energy_uste")
    zeta_cor = float(zeta_cor2) + ((a2 ** -3) / denom) * (float(zeta_cor2) - float(zeta_cor1))
    return zeta_cor


def correlation_energy_uspe(zeta_HF: float, zeta_E: float) -> float:
    """
    USPE-style single-basis correlation contribution.
    Returns zeta_Ecr = zeta_E - zeta_HF
    """
    return float(zeta_E) - float(zeta_HF)


# ----------------------------
# Flexible CBS_extrapolation dispatcher
# ----------------------------
def CBS_extrapolation(*args, **kwargs) -> Any:
    """
    Flexible dispatcher for CBS extrapolation.

    Supports USTE-style and USPE-style calls.
    Prefer explicit exact-length detection to avoid misclassification.
    """
    # If exactly 5 positional args => USPE-style (HF, E, method, constant, basis)
    if len(args) == 5:
        try:
            zeta_HF, zeta_E, method, constant, basis = args[:5]
            _, dic_corr = dictionaries(method, basis, None)
            a_values = select_constant(constant)
            zeta_Ecr = correlation_energy_uspe(zeta_HF, zeta_E)
            a_val = float(a_values.get(method))
            denom = float(dic_corr[basis]) ** 3
            zeta_total = float(zeta_Ecr) + (a_val * float(zeta_E) / denom)
            return float(zeta_total)
        except Exception as e:
            raise TypeError(f"[tensorial_properties1] USPE-style CBS_extrapolation failed: {e}")

    # USTE-style detection: require at least 7 args (HF1, HF2, cor1, cor2, dic, basis1, basis2)
    if len(args) >= 7:
        try:
            zeta_HF1, zeta_HF2, zeta_cor1, zeta_cor2, dic_correlacion, basis1, basis2 = args[:7]
            zeta_HF = hartree_fock_energy(zeta_HF1, zeta_HF2, basis1, basis2)
            zeta_cor = dynamic_correlation_energy_uste(zeta_cor1, zeta_cor2, dic_correlacion, basis1, basis2)
            zeta_total = float(zeta_HF) + float(zeta_cor)
            return zeta_HF, zeta_cor, zeta_total
        except Exception as e:
            raise TypeError(f"[tensorial_properties1] USTE-style CBS_extrapolation failed: {e}")

    # Nothing matched
    raise TypeError("[tensorial_properties1] CBS_extrapolation: unsupported arguments; "
                    "expected either USPE-style (5 args) or USTE-style (7+ args).")

