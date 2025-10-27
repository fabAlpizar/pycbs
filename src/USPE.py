import math
from basis import dc1, dc2, dc3  # Import dictionaries form basis.py

# Function to select the dictionary and the constant value
def dictionaries(method, basis):
    """Return the correlation dictionary and the constant based on the selected method."""
    if method == "MP2":
        return dc1
    elif method == "CCSD(T)":
        return dc2
    elif method == "MP2+CCSD(T)":
        return dc3
    else:
        raise ValueError("Invalid Method!!!")

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

# Correlation enegy calculation function
def correlation_energy(HF, Etot):
    """Calculate correlation energy."""
    Ecr = Etot - HF
    return Ecr

# CBS extrapolation energy calculation function
def CBS_extrapolation(HF, Etot, method, constant, basis):
    """Calculate CBS extrapolation energy."""
    dic_correlation = dictionaries(method, basis)  
    a_values = select_constant(constant)  
    Ecr = correlation_energy(HF, Etot)  
    CBS = Ecr + ((a_values[method] * Etot) / dic_correlation[basis]**3)
    return CBS





