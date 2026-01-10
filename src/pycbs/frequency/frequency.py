import math
from pycbs.basis import hf, dc1, dc2, dc3  # Import dictionaries form basis.py

# Function to retrieve dictionaries based on the selected method and two chosen databases
def dictionaries(metodo, basis1, basis2):
    """"Return Hartree-Fock and correlation dictionaries based on the selected method and bases."""
    if metodo == "MP2":
        dic_correlacion = dc1
    elif metodo == "CCSD(T)":
        dic_correlacion = dc2
    elif metodo == "MP2+CCSD(T)":
        dic_correlacion = dc3
    else:
        raise ValueError("Invalid Method!!!")

    # Validate database existence
    for basis in [basis1, basis2]:
        if basis not in hf:
            raise ValueError(f"The basis set '{basis}' is not recognized.")
        if basis not in dic_correlacion:
            raise ValueError(f"The basis set '{basis}' is not recognized.")

    return hf, dic_correlacion

# Correlation frequency calculation function
def correlation_frequency(HF1, HF2, F1, F2):
    """Calculate the correlation energy."""
    correlation_frequency1 = F1 - HF1
    correlation_frequency2 = F2 - HF2
    return correlation_frequency1, correlation_frequency2

# Hartree-Fock extrapolation function
def hartree_fock_frequency(HF1, HF2, basis1, basis2):
    """Calculate the static correlation."""
    EHF = (HF1 * math.exp(2.284 * hf[basis1]) - HF2 * math.exp(2.284 * hf[basis2])) / \
          (math.exp(2.284 * hf[basis1]) - math.exp(2.284 * hf[basis2]))
    return EHF

# Dynamic correlation calculation function
def dynamic_correlation_frequency(Fcr1, Fcr2, dic_correlacion, basis1, basis2):
    """Calculate the dynamic correlation."""
    dc = Fcr2 + ((dic_correlacion[basis2] ** (-3)) / ((dic_correlacion[basis1] ** (-3)) - (dic_correlacion[basis2] ** (-3)))) * (Fcr2 - Fcr1)
    return dc

# CBS extrapolation calculation function
def CBS_extrapolation(HF1, HF2, Fcr1, Fcr2, dic_correlacion, basis1, basis2):
    """Calculate the CBS extrapolation energy."""
    EHF = hartree_fock_frequency(HF1, HF2, basis1, basis2)
    dc = dynamic_correlation_frequency(Fcr1, Fcr2, dic_correlacion, basis1, basis2)
    CBS = EHF + dc
    return EHF, dc, CBS

