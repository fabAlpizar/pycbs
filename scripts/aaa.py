#!/usr/bin/env python3
"""
Calculate single-point energy of a molecule using PySCF.
Supports HF, MP2, and CCSD(T) methods.
"""
from pathlib import Path
import sys

try:
    from pyscf import gto, scf, cc, mp, lib
except ImportError:
    print("Error: PySCF is required. Install with: pip install pyscf")
    sys.exit(1)

# ===========================
# CONFIGURATION
# ===========================
XYZ_FILE = "/home/fab/pyCBS/PyCBS-OUTPUTS/h2co_cycle_001.xyz"  # ← CHANGE THIS TO YOUR XYZ FILE
METHOD = "CCSD(T)"  # Options: "HF", "MP2", "CCSD(T)"
BASIS_SET = "cc-pvtz"  # Basis set to use
SPIN = 0  # Multiplicity (0 for closed shell, >0 for open shell)
CHARGE = 0  # Molecular charge

# PySCF settings
PYSCF_THREADS = 4
lib.num_threads(PYSCF_THREADS)


# ===========================
# HELPER FUNCTIONS
# ===========================
def read_xyz(filename):
    """Read XYZ file and return atoms, coordinates, and comment"""
    with open(filename) as f:
        natoms = int(f.readline().strip())
        comment = f.readline().rstrip('\n')
        atoms = []
        coords = []
        for _ in range(natoms):
            parts = f.readline().split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, coords, comment


def xyz_to_pyscf_string(atoms, coords):
    """Convert atoms and coordinates to PySCF format string"""
    lines = []
    for atom, coord in zip(atoms, coords):
        lines.append(f"{atom} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}")
    return "\n".join(lines)


def calculate_energy(atoms, coords, method, basis, spin, charge):
    """
    Calculate energy using PySCF

    Parameters:
    -----------
    atoms : list of str
        Atomic symbols
    coords : list of list
        Cartesian coordinates
    method : str
        "HF", "MP2", or "CCSD(T)"
    basis : str
        Basis set (e.g., "cc-pvtz", "def2-svp", "sto-3g")
    spin : int
        Multiplicity (0 for closed shell)
    charge : int
        Molecular charge

    Returns:
    --------
    dict with energy data
    """
    # Create PySCF molecule object
    mol = gto.Mole()
    mol.atom = xyz_to_pyscf_string(atoms, coords)
    mol.basis = basis
    mol.spin = int(spin)
    mol.charge = int(charge)
    mol.nthread = PYSCF_THREADS
    mol.max_memory = 8000  # MB
    mol.verbose = 3  # Print output
    mol.build()

    print(f"\n{'=' * 80}")
    print(f"SINGLE POINT ENERGY CALCULATION")
    print(f"{'=' * 80}")
    print(f"Method: {method}")
    print(f"Basis set: {basis}")
    print(f"Spin multiplicity: {spin}")
    print(f"Charge: {charge}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Number of electrons: {mol.nelectron}")
    print(f"Number of basis functions: {mol.nbas}")
    print(f"{'=' * 80}\n")

    # HF-SCF calculation
    print(f"Computing RHF-SCF...")
    mf = scf.RHF(mol)
    mf.max_memory = 14330
    mf.conv_tol = 1e-9
    mf.max_cycle = 100
    scf_energy = mf.kernel()

    scf_converged = getattr(mf, "converged", True)
    if not scf_converged:
        print("WARNING: SCF did not fully converge!")

    results = {
        'method': method,
        'basis': basis,
        'scf_energy': float(scf_energy),
    }

    # Correlation energy calculation
    if method.upper() == "HF":
        print(f"\nFinal HF energy: {scf_energy:.10f} Ha")
        results['total_energy'] = float(scf_energy)
        results['correlation_energy'] = 0.0

    elif method.upper() == "MP2":
        print(f"Computing MP2...")
        mymp = mp.MP2(mf)
        mymp.max_memory = 14330
        res = mymp.run()

        mp2_total = getattr(res, 'e_tot', None)
        if mp2_total is None:
            mp2_total = getattr(mymp, 'e_tot', None)
        if mp2_total is None:
            e_corr = getattr(res, 'e_corr', getattr(mymp, 'e_corr', None))
            if e_corr is not None:
                mp2_total = scf_energy + e_corr

        if mp2_total is None:
            raise RuntimeError("Could not retrieve MP2 total energy")

        mp2_total = float(mp2_total)
        mp2_corr = mp2_total - scf_energy

        print(f"HF energy: {scf_energy:.10f} Ha")
        print(f"MP2 correlation energy: {mp2_corr:.10f} Ha")
        print(f"Final MP2 energy: {mp2_total:.10f} Ha")

        results['total_energy'] = float(mp2_total)
        results['correlation_energy'] = float(mp2_corr)

    elif method.upper() == "CCSD(T)":
        print(f"Computing CCSD...")
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1e-7
        mycc.max_cycle = 100
        mycc.kernel()

        ccsd_total = mycc.e_tot
        ccsd_corr = ccsd_total - scf_energy

        print(f"Computing (T) correction...")
        try:
            et = mycc.ccsd_t()
        except Exception as e:
            print(f"Warning: Could not compute (T) correction: {e}")
            et = 0.0

        et = et if et is not None else 0.0
        ccsd_t_total = ccsd_total + et

        print(f"HF energy: {scf_energy:.10f} Ha")
        print(f"CCSD correlation energy: {ccsd_corr:.10f} Ha")
        print(f"(T) correction: {et:.10f} Ha")
        print(f"Final CCSD(T) energy: {ccsd_t_total:.10f} Ha")

        results['total_energy'] = float(ccsd_t_total)
        results['ccsd_energy'] = float(ccsd_total)
        results['correlation_energy'] = float(ccsd_corr)
        results['triples_correction'] = float(et)

    else:
        raise ValueError(f"Unknown method: {method}")

    return results


def print_summary(results):
    """Print summary of results"""
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Method: {results['method']}")
    print(f"Basis set: {results['basis']}")
    print(f"SCF energy: {results['scf_energy']:.10f} Ha")

    if results['method'].upper() == "HF":
        print(f"Total energy: {results['total_energy']:.10f} Ha")

    elif results['method'].upper() == "MP2":
        print(f"MP2 correlation energy: {results['correlation_energy']:.10f} Ha")
        print(f"Total MP2 energy: {results['total_energy']:.10f} Ha")

    elif results['method'].upper() == "CCSD(T)":
        print(f"CCSD correlation energy: {results['correlation_energy']:.10f} Ha")
        print(f"(T) correction: {results['triples_correction']:.10f} Ha")
        print(f"Total CCSD(T) energy: {results['total_energy']:.10f} Ha")

    print(f"{'=' * 80}\n")


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    # Check if file exists
    if not Path(XYZ_FILE).exists():
        print(f"Error: File '{XYZ_FILE}' not found!")
        print(f"Please update XYZ_FILE variable in the script.")
        sys.exit(1)

    # Read XYZ file
    print(f"Reading {XYZ_FILE}...")
    atoms, coords, comment = read_xyz(XYZ_FILE)
    print(f"Loaded {len(atoms)} atoms")
    print(f"Comment: {comment}\n")

    # Calculate energy
    try:
        results = calculate_energy(atoms, coords, METHOD, BASIS_SET, SPIN, CHARGE)
        print_summary(results)
    except Exception as e:
        print(f"\nError during calculation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)