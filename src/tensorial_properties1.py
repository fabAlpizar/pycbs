import math
from basis import hf, dc1, dc2, dc3  # Import dictionaries form basis.py
zeta = "ζ"

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

# Function to select the constant value
def select_constant(constant):
    """Select the constant value."""
    a1 = {'MP2': 0.0111, 'CCSD': 0.0073, 'CCSD(T)': 0.0078}
    a2 = {'MP2': 0.0094, 'CCSD': 0.0061, 'CCSD(T)': 0.0065}

    if constant == "normal":
        return a1
    elif constant == "augmented":
        return a2
    else:
        raise ValueError("Invalid constat!!!.")


# Correlation energy calculation function
def correlation_energy(zeta_HF1, zeta_HF2, zeta_E1, zeta_E2):
    """Calculate the correlation energy."""
    zeta_cor1 = zeta_E1 - zeta_HF1
    zeta_cor2 = zeta_E2 - zeta_HF2
    return zeta_cor1, zeta_cor2

# Hartree-Fock extrapolation function 
def hartree_fock_energy(zeta_HF1, zeta_HF2, basis1, basis2):
    """Calculate the static correlation."""
    zeta_HF = zeta_HF2 +  ((math.exp(2.284*hf[basis1])) / (math.exp(2.284*hf[basis2])-math.exp(2.284*hf[basis1])))*(zeta_HF2 - zeta_HF1)
    return zeta_HF

# Dynamic correlation calculation function
def dynamic_correlation_energy(zeta_cor1, zeta_cor2, dic_correlacion, basis1, basis2):
    """Calculate the dynamic correlation."""
    zeta_cor = zeta_cor2 + ((dic_correlacion[basis2] ** (-3)) / ((dic_correlacion[basis1] ** (-3)) - (dic_correlacion[basis2] ** (-3)))) * (zeta_cor2 - zeta_cor1)
    return zeta_cor

# CBS extrapolation calculation function
def CBS_extrapolation(zeta_HF1, zeta_HF2, zeta_cor1, zeta_cor2, dic_correlacion, basis1, basis2):
    """Calculate the CBS extrapolation energy."""
    zeta_HF = hartree_fock_energy(zeta_HF1, zeta_HF2, basis1, basis2)
    zeta_cor = dynamic_correlation_energy(zeta_cor1, zeta_cor2, dic_correlacion, basis1, basis2)
    zeta = zeta_HF + zeta_cor
    return zeta_HF, zeta_cor, zeta


#uspe scheme

# Correlation enegy calculation function
def correlation_energy(zeta_HF, zeta_E):
    """Calculate correlation energy."""
    zeta_Ecr = zeta_E - zeta_HF
    return zeta_Ecr

# CBS extrapolation energy calculation function
def CBS_extrapolation(zeta_HF, zeta_E, method, constant, basis):
    """Calculate CBS extrapolation energy."""
    dic_correlation = dictionaries(method, basis)  
    a_values = select_constant(constant)  
    zeta_Ecr = correlation_energy(zeta_HF, zeta_E)  
    zeta = zeta_Ecr + ((a_values[method] * zeta_E) / dic_correlation[basis]**3)
    return zeta

