import math
from src.pycbs.basis import hf, dc1, dc2, dc3  # Import dictionaries form basis.py

def dictionaries(metodo, basis1, basis2, basis3, basis4):
    """"Return Hartree-Fock and correlation dictionaries based on the selected method and bases."""
    if metodo == "MP2":
        dic_correlacion = dc1
    elif metodo == "CCSD(T)":
        dic_correlacion = dc2
    elif metodo == "MP2+CCSD(T)":
        dic_correlacion = dc3
    else:
        raise ValueError("Invalid Method!!!")

    # Validate HF existence
    for basis in [basis1, basis2]:
        if basis not in hf:
            raise ValueError(f"The basis set '{basis}' is not recognized.")
    
    # Validate dc existence
    for basis in [basis3, basis4]:
        if basis not in dic_correlacion:
            raise ValueError(f"The basis set '{basis}' is not recognized.")

    return hf, dic_correlacion


# Correlation energy calculation function
def correlation_energy(HF1, HF2, E1, E2):
    """Calculate the correlation energy."""
    correlation_energy1 = E1 - HF1
    correlation_energy2 = E2 - HF2
    return correlation_energy1, correlation_energy2

# Hartree-Fock extrapolation function 
def hartree_fock_energy(HF1, HF2, basis1, basis2):
    """Calculate the static correlation."""
    EHF = (HF1 * math.exp(2.284 * hf[basis1]) - HF2 * math.exp(2.284 * hf[basis2])) / \
          (math.exp(2.284 * hf[basis1]) - math.exp(2.284 * hf[basis2]))
    return EHF

# Dynamic correlation calculation function
def dynamic_correlation_energy(Ecr1, Ecr2, dic_correlacion, basis3, basis4):
    """Calculate the dynamic correlation."""
    dc = Ecr2 + ((dic_correlacion[basis4] ** (-3)) / ((dic_correlacion[basis3] ** (-3)) - (dic_correlacion[basis4] ** (-3)))) * (Ecr2 - Ecr1)
    return dc

# CBS extrapolation calculation function
def CBS_extrapolation(HF1, HF2, Ecr1, Ecr2, dic_correlacion, basis1, basis2, basis3, basis4):
    """Calculate the CBS extrapolation energy."""
    EHF = hartree_fock_energy(HF1, HF2, basis1, basis2)
    dc = dynamic_correlation_energy(Ecr1, Ecr2, dic_correlacion, basis3, basis4)
    CBS = EHF + dc
    return EHF, dc, CBS



