#!/usr/bin/env python3
"""
opt_3.py — CBS geometry optimizer using redundant internal coordinates + L-BFGS

Usage (from shell):
    python opt_3.py input.xyz --output optimized.xyz

Requirements:
    - numpy
    - scipy
    - pyscf

Design goals / notes:
    - compute_cbs_energy returns SCF energy and correlation energy separately
      and then the CBS composition is computed from those components.
    - internal_to_cartesian uses least_squares to reconstruct Cartesian coords
      from target internals (robust numerical inversion).
    - caching of energy evaluations to reduce redundant PySCF calculations.
    - default basis set pair: cc-pVDZ / cc-pVTZ and correlation extrapolation ~ n^-3.
      These are configurable in the CONFIG block below.
"""
import argparse
import math
import copy
import sys
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize, least_squares

# PySCF imports
try:
    from pyscf import gto, scf, mp, cc
except Exception as e:
    raise ImportError("PySCF is required for this script. Install pyscf and retry.") from e


# ---------------------
# CONFIGURATION
# ---------------------
CONFIG = {
    # Basis sets: smaller first, larger second
    "BASIS_SETS": ("cc-pvdz", "cc-pvtz"),

    # CBS extrapolation parameters:
    # correlation extrapolation using power law ~ X^{-3} (we use X1=2, X2=3 for DZ/TZ)
    "X1_CORR": 2.00,
    "X2_CORR": 3.00,
    # HF extrapolation model uses an exponential; BETA often ~ 1.60
    "X1_HF": 2.00,
    "X2_HF": 3.00,
    "BETA_HF": 1.63,

    # PySCF settings
    "PYSCF_MAX_MEMORY": 4 * 1024,  # in MB
    "PYSCF_NTHREADS": 1,

    # Geometry inversion (least_squares) options
    "LSQRTOL": 1e-8,
    "LSQRMAXITER": 200,

    # Optimization (L-BFGS-B) options
    "ENERGY_TOL": 1e-8,    # Hartree stopping criterion for CBS energy change
    "X_TOL": 1e-4,         # step tolerance for internal coordinates
    "LBFGSB_MAXITER": 200,
    "BOND_MIN_FACTOR": 0.5,  # lower bound for bond length relative to initial
    "BOND_MAX_FACTOR": 2.0,  # upper bound for bond length relative to initial
}


# ---------------------
# Utility geometry functions
# ---------------------
def read_xyz(filename):
    """
    Read a simple XYZ file. Returns (symbols, coords) where coords is an (N,3) float array.
    """
    with open(filename) as fh:
        lines = [l.rstrip() for l in fh if l.strip()]
    natom = int(lines[0].split()[0])
    header = lines[1]
    data = lines[2:2 + natom]
    syms = []
    coords = []
    for line in data:
        parts = line.split()
        syms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return syms, np.array(coords)


def write_xyz(filename, symbols, coords, comment=""):
    nat = len(symbols)
    with open(filename, "w") as fh:
        fh.write(f"{nat}\n{comment}\n")
        for s, c in zip(symbols, coords):
            fh.write(f"{s:2s} {c[0]: .10f} {c[1]: .10f} {c[2]: .10f}\n")


def dist(a, b):
    return np.linalg.norm(a - b)


def angle(a, b, c):
    # angle at b given positions a-b-c
    ba = a - b
    bc = c - b
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosv = np.clip(cosv, -1.0, 1.0)
    return math.acos(cosv)


def dihedral(a, b, c, d):
    # returns signed dihedral angle between planes (a-b-c) and (b-c-d)
    b1 = a - b
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1u = n1 / (np.linalg.norm(n1) + 1e-16)
    n2u = n2 / (np.linalg.norm(n2) + 1e-16)
    m1 = np.cross(n1u, (b2 / (np.linalg.norm(b2) + 1e-16)))
    x = np.dot(n1u, n2u)
    y = np.dot(m1, n2u)
    return math.atan2(y, x)


# ---------------------
# Build redundant internals
# ---------------------
def build_internals(symbols, coords, cutoff_factor=1.2):
    """
    Build a list of internal coordinate descriptors for a molecule:
      - bonds: (i, j)
      - angles: (i, j, k)
      - dihedrals: (i, j, k, l)

    For robust general use we include all bonds whose distance is less than cutoff_factor * covalent_sum,
    where covalent radii are estimated by a simple table (defaults otherwise).
    """
    # Simple covalent radii table (Å). Extend if needed.
    covrad = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02,
    }

    nat = len(symbols)
    coords = np.asarray(coords)
    bonds = []
    for i in range(nat):
        for j in range(i + 1, nat):
            rsum = covrad.get(symbols[i], 0.8) + covrad.get(symbols[j], 0.8)
            if dist(coords[i], coords[j]) <= cutoff_factor * rsum:
                bonds.append((i, j))

    # angles: any connected triplet i-j-k where bonds (i, j) and (j, k)
    angles = []
    bond_set = set(bonds)
    adjacency = {i: set() for i in range(nat)}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for j in range(nat):
        for i in adjacency[j]:
            for k in adjacency[j]:
                if i < k:
                    angles.append((i, j, k))

    # dihedrals: i-j-k-l where bonds (i, j), (j, k), (k, l)
    dihedrals = []
    for (i, j) in bonds:
        for k in adjacency[j]:
            if k == i:
                continue
            for l in adjacency[k]:
                if l == j:
                    continue
                # ensure uniqueness ordering
                if i < l:
                    dihedrals.append((i, j, k, l))

    # canonical internal descriptor: a list of dicts with 'type' and indices
    internals = []
    for (i, j) in bonds:
        internals.append({"type": "bond", "idx": (i, j)})
    for (i, j, k) in angles:
        internals.append({"type": "angle", "idx": (i, j, k)})
    for (i, j, k, l) in dihedrals:
        internals.append({"type": "dihedral", "idx": (i, j, k, l)})

    # compute initial numeric values
    values = internals_to_values(internals, coords)
    return internals, np.array(values)


def internals_to_values(internals, coords):
    values = []
    for it in internals:
        tp = it["type"]
        idx = it["idx"]
        if tp == "bond":
            i, j = idx
            values.append(dist(coords[i], coords[j]))
        elif tp == "angle":
            i, j, k = idx
            values.append(angle(coords[i], coords[j], coords[k]))
        elif tp == "dihedral":
            i, j, k, l = idx
            values.append(dihedral(coords[i], coords[j], coords[k], coords[l]))
        else:
            raise ValueError("Unknown internal type: " + tp)
    return values


# ---------------------
# Internal -> Cartesian inversion via least squares
# ---------------------
def internal_to_cartesian(initial_cart, internals, target_values, lsq_opts=None):
    """
    Given an initial Cartesian guess (N x 3), find Cartesian coordinates whose internals
    match target_values as closely as possible by minimizing residuals with least_squares.

    Returns the flattened cartesian vector (3N).
    """
    nat = initial_cart.shape[0]
    x0 = initial_cart.reshape(-1)

    if lsq_opts is None:
        lsq_opts = {}
    tol = lsq_opts.get("tol", CONFIG["LSQRTOL"])
    max_nfev = lsq_opts.get("max_nfev", CONFIG["LSQRMAXITER"])

    def residuals(x):
        coords = x.reshape((nat, 3))
        vals = internals_to_values(internals, coords)
        return np.array(vals) - np.array(target_values)

    # Use least_squares with robust method 'trf' or 'dogbox' which handle bounds; no bounds here
    result = least_squares(residuals, x0, method="trf", ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)
    if not result.success:
        # still return the best found solution, but warn
        print("Warning: internal->cartesian least_squares did not converge: " + result.message, file=sys.stderr)
    return result.x.reshape((nat, 3))


# ---------------------
# CBS composition
# ---------------------
def cbs_compose(corr_small, corr_big, scf_small, scf_big, cfg=CONFIG):
    """
    Compose extrapolated CBS energy from:
      corr_small, corr_big : correlation energies at basis sets 1 and 2
      scf_small, scf_big   : SCF energies at basis sets 1 and 2

    Uses the standard separation: HF extrapolated with exponential, correlation with power law n^-3.
    """
    X1 = cfg["X1_CORR"]
    X2 = cfg["X2_CORR"]
    X1_HF = cfg["X1_HF"]
    X2_HF = cfg["X2_HF"]
    BETA = cfg["BETA_HF"]

    # correlation extrapolation coefficient using power -3
    A_corr = (X1 ** 3) / (X2 ** 3 - X1 ** 3)

    # HF exponential coefficient
    denom = math.exp(BETA * X2_HF) - math.exp(BETA * X1_HF)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    B_hf = math.exp(BETA * X1_HF) / denom

    # Extrapolate
    E_corr_cbs = corr_big + A_corr * (corr_big - corr_small)
    E_hf_cbs = scf_big + (math.exp(BETA * X2_HF) * (scf_big - scf_small)) / denom - (math.exp(BETA * X1_HF) * (scf_big - scf_small)) / denom
    # The algebra above is equivalent to the common form; combine:
    # But to match previous code style, use a clear expression:
    E_cbs = E_corr_cbs + E_hf_cbs
    return float(E_cbs), float(E_hf_cbs), float(E_corr_cbs)


# ---------------------
# Energy evaluation with PySCF (returns scf_energy, corr_energy)
# ---------------------
def compute_scf_and_correlation(symbols, coords, basis, method_preference=("ccsd(t)", "mp2")):
    """
    Compute SCF energy and correlation energy for a given basis:
      - returns (scf_energy, corr_energy)
    Tries CCSD(T) first (if in preference) and falls back to MP2 if needed.
    """
    nat = len(symbols)
    # build PySCF molecule
    mol = gto.Mole()
    mol.verbose = 0
    mol.unit = "Angstrom"
    mol.atom = [(symbols[i], tuple(coords[i].tolist())) for i in range(nat)]
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.build()

    # configure resources
    mol.max_memory = CONFIG["PYSCF_MAX_MEMORY"]
    mol.stdout = None
    # set thread: pyscf uses environment var or internal; don't set global here

    # SCF (RHF assumed)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-8
    mf.verbose = 0
    mf.max_cycle = 200
    try:
        scf_e = mf.kernel()
    except Exception as e:
        raise RuntimeError("SCF failed: " + str(e))

    # attempt correlation methods
    corr_e = None
    last_err = None
    if "ccsd(t)" in method_preference:
        try:
            mycc = cc.CCSD(mf)
            mycc.conv_tol = 1e-7
            mycc.max_cycle = 200
            mycc.kernel()
            triples = 0.0
            try:
                triples = mycc.ccsd_t()
            except Exception:
                triples = 0.0
            # prefer explicit correlation attribute
            ccsd_corr = getattr(mycc, "e_corr", None)
            if ccsd_corr is None:
                # fallback: total - scf
                ccsd_corr = float(getattr(mycc, "e_tot", scf_e) - scf_e)
            corr_e = float(ccsd_corr + (triples if triples is not None else 0.0))
        except Exception as e:
            last_err = e
            corr_e = None

    if corr_e is None and "mp2" in method_preference:
        try:
            mp2 = mp.MP2(mf)
            mp2.verbose = 0
            res = mp2.kernel()
            # prefer res.e_corr, then mp2.e_corr, then res.e_tot - scf
            mp2_corr = None
            if hasattr(res, "e_corr"):
                mp2_corr = float(res.e_corr)
            elif hasattr(mp2, "e_corr"):
                mp2_corr = float(mp2.e_corr)
            elif hasattr(res, "e_tot"):
                mp2_corr = float(res.e_tot - scf_e)
            else:
                mp2_corr = float(res - scf_e) if isinstance(res, (float, int)) else None
            if mp2_corr is None:
                raise RuntimeError("Could not retrieve MP2 correlation energy from PySCF objects")
            corr_e = float(mp2_corr)
        except Exception as e:
            last_err = e
            corr_e = None

    if corr_e is None:
        raise RuntimeError("No correlation energy available for basis {} (last error: {})".format(basis, last_err))

    return float(scf_e), float(corr_e)


# ---------------------
# High-level CBS energy: evaluate molecule at two basis sets and return CBS energy
# ---------------------
def compute_cbs_energy(symbols, coords, basis_pair=None, cfg=CONFIG):
    """
    Compute CBS-extrapolated energy for given Cartesian coordinates.
    Returns: (E_CBS, E_HF_CBS, E_CORR_CBS, debug_dict)
    """
    if basis_pair is None:
        basis_pair = cfg["BASIS_SETS"]
    # small basis first
    bs1, bs2 = basis_pair

    scf1, corr1 = compute_scf_and_correlation(symbols, coords, bs1)
    scf2, corr2 = compute_scf_and_correlation(symbols, coords, bs2)

    E_cbs, E_hf_cbs, E_corr_cbs = cbs_compose(corr1, corr2, scf1, scf2, cfg=cfg)
    debug = {
        "scf1": scf1, "scf2": scf2,
        "corr1": corr1, "corr2": corr2,
        "E_hf_cbs": E_hf_cbs, "E_corr_cbs": E_corr_cbs
    }
    return E_cbs, E_hf_cbs, E_corr_cbs, debug


# ---------------------
# Caching wrapper for energy evaluations (keyed by rounded internal coordinates)
# ---------------------
class EnergyCache:
    def __init__(self, symbols, internals, coords0, rounding=8, cfg=None):
        self.symbols = symbols
        self.internals = internals
        self.coords0 = np.array(coords0)
        self.rounding = rounding
        # cfg is a dict-like configuration controlling extrapolation formula
        self.cfg = cfg if cfg is not None else CONFIG
        self._cache = {}

    def _key_from_internals(self, internal_vector):
        # Round internal vector to given digits to make the key stable
        return tuple([round(float(x), self.rounding) for x in internal_vector])

    def evaluate(self, internal_vector, basis_pair=None):
        k = self._key_from_internals(internal_vector)
        if k in self._cache:
            return self._cache[k]
        # reconstruct Cartesian
        cart = internal_to_cartesian(self.coords0, self.internals, internal_vector)
        # compute CBS using the cached cfg
        E_cbs, E_hf_cbs, E_corr_cbs, debug = compute_cbs_energy(self.symbols, cart, basis_pair, cfg=self.cfg)
        res = {"E_cbs": E_cbs, "E_hf_cbs": E_hf_cbs, "E_corr_cbs": E_corr_cbs, "cart": cart, "debug": debug}
        self._cache[k] = res
        return res



# ---------------------
# Optimization driver
# ---------------------
def optimize_geometry(symbols, coords0, basis_pair=None, options=None):
    """
    Optimize geometry (internals) with L-BFGS-B, using CBS energy as objective.
    Returns optimized carts, final energy, and result dict.
    """
    opt_history = []
    cycle_counter = {"i": 0}

    if options is None:
        options = {}

    # Build a working copy of CONFIG and override with options if provided.
    # Options keys expected: X1, X2, Xhf1, Xhf2 (floats). Map them into internal names.
    cfg = copy.deepcopy(CONFIG)
    # map option names to CONFIG keys
    if "X1" in options and options["X1"] is not None:
        cfg["X1_CORR"] = float(options["X1"])
    if "X2" in options and options["X2"] is not None:
        cfg["X2_CORR"] = float(options["X2"])
    if "Xhf1" in options and options["Xhf1"] is not None:
        cfg["X1_HF"] = float(options["Xhf1"])
    if "Xhf2" in options and options["Xhf2"] is not None:
        cfg["X2_HF"] = float(options["Xhf2"])

    internals, values0 = build_internals(symbols, coords0)
    # pass cfg to the EnergyCache so compute_cbs_energy uses these parameters
    energy_cache = EnergyCache(symbols, internals, coords0, rounding=8, cfg=cfg)


    # objective function for scipy minimize: returns scalar CBS energy
    def objective(x):
        res = energy_cache.evaluate(x, basis_pair=basis_pair)
        # record per-evaluation history (cycle index + CBS energy)
        E_cbs = res["E_cbs"]
        cycle_counter["i"] += 1
        opt_history.append({
            "cycle": int(cycle_counter["i"]),
            "cbs_energy": float(E_cbs),
        })
        return res["E_cbs"]

    # provide a simple callback to mirror progress (not too verbose)
    last = {"E": None}
    def callback(xk):
        E = energy_cache.evaluate(xk, basis_pair=basis_pair)["E_cbs"]
        if last["E"] is None or abs(E - last["E"]) > 1e-8:
            print(f"opt step: E_cbs = {E:.10f} Ha")
            last["E"] = E

    x0 = np.array(values0, dtype=float)

    # run minimizer (L-BFGS-B). Note: we don't provide analytic gradient here; scipy will approximate via finite differences.
    opt_res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": cfg["ENERGY_TOL"],
            "maxiter": cfg["LBFGSB_MAXITER"],
            "maxls": 20,
            "gtol": 1e-8,
            "eps": 1e-6,
        },
        callback=callback,
    )

    x_opt = opt_res.x
    final = energy_cache.evaluate(x_opt, basis_pair=basis_pair)
    final_cart = final["cart"]
    result = {
        "opt_result": opt_res,
        "final_energy": final["E_cbs"],
        "final_hf_cbs": final["E_hf_cbs"],
        "final_corr_cbs": final["E_corr_cbs"],
        "final_cart": final_cart,
        "debug": final["debug"],
        "internals": internals,
        "x_opt": x_opt,
        "history": opt_history,
    }
    return result


# ---------------------
# CLI
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="CBS geometry optimizer (internals + L-BFGS).")
    parser.add_argument("input_xyz", help="Input XYZ file")
    parser.add_argument("--output", "-o", default="opt_out.xyz", help="Output optimized XYZ")
    parser.add_argument("--basis1", default=None, help="Smaller basis set override")
    parser.add_argument("--basis2", default=None, help="Larger basis set override")
    args = parser.parse_args()

    symbols, coords0 = read_xyz(args.input_xyz)
    basis_pair = None
    if args.basis1 and args.basis2:
        basis_pair = (args.basis1, args.basis2)
    else:
        basis_pair = CONFIG["BASIS_SETS"]

    print("Starting optimization with basis pair:", basis_pair)
    res = optimize_geometry(symbols, coords0, basis_pair=basis_pair)
    write_xyz(args.output, symbols, res["final_cart"], comment=f"CBS opt energy {res['final_energy']:.10f} Ha")
    print("Optimization finished. Final CBS energy (Ha):", res["final_energy"])
    print("Wrote optimized geometry to:", args.output)
    # You may print debug info if desired:
    # print(res["debug"])


if __name__ == "__main__":
    main()
