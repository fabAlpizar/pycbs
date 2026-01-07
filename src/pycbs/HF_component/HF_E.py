import math
from src.basis import hf
def hartree_fock_energy(HF1, HF2, basis1, basis2):
    """Calculate the static correlation."""
    EHF = (HF1 * math.exp(2.284 * hf[basis1]) - HF2 * math.exp(2.284 * hf[basis2])) / \
          (math.exp(2.284 * hf[basis1]) - math.exp(2.284 * hf[basis2]))
    return EHF