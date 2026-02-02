#!/usr/bin/env python3
"""
opt_3.py — CBS geometry optimizer using redundant internal coordinates + L-BFGS
"""
import argparse
import math
import sys
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
    "BASIS_SETS": ("cc-pvdz", "cc-pvtz"),
    "X1_CORR": 2.00,
    "X2_CORR": 3.00,
    "X1_HF": 2.00,
    "X2_HF": 3.00,
    "BETA_HF": 1.63,
    "PYSCF_MAX_MEMORY": 4 * 1024,
    "PYSCF_NTHREADS": 1,
    "LSQRTOL": 1e-8,
    "LSQRMAXITER": 200,
    "ENERGY_TOL": 1e-8,
    "X_TOL": 1e-4,
    "LBFGSB_MAXITER": 200,
    "BOND_MIN_FACTOR": 0.5,
    "BOND_MAX_FACTOR": 2.0,
}


# ---------------------
# XYZ I/O
# ---------------------
def read_xyz(filename):
    with open(filename) as fh:
        lines = [l.rstrip() for l in fh if l.strip()]
    nat = int(lines[0])
    data = lines[2:2 + nat]
    syms, coords = [], []
    for line in data:
        p = line.split()
        syms.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, np.array(coords)


def write_xyz(filename, symbols, coords, comment=""):
    with open(filename, "w") as fh:
        fh.write(f"{len(symbols)}\n{comment}\n")
        for s, c in zip(symbols, coords):
            fh.write(f"{s:2s} {c[0]: .10f} {c[1]: .10f} {c[2]: .10f}\n")


# ---------------------
# Geometry helpers
# ---------------------
def dist(a, b):
    return np.linalg.norm(a - b)


def angle(a, b, c):
    ba, bc = a - b, c - b
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return math.acos(np.clip(cosv, -1.0, 1.0))


def dihedral(a, b, c, d):
    b1, b2, b3 = a - b, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    n1 /= np.linalg.norm(n1) + 1e-16
    n2 /= np.linalg.norm(n2) + 1e-16
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-16))
    return math.atan2(np.dot(m1, n2), np.dot(n1, n2))


# ---------------------
# Internals
# ---------------------
def build_internals(symbols, coords, cutoff_factor=1.2):
    covrad = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
    nat = len(symbols)
    bonds = []

    for i in range(nat):
        for j in range(i + 1, nat):
            rsum = covrad.get(symbols[i], 0.8) + covrad.get(symbols[j], 0.8)
            if dist(coords[i], coords[j]) <= cutoff_factor * rsum:
                bonds.append((i, j))

    adjacency = {i: set() for i in range(nat)}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)

    internals = []
    for i, j in bonds:
        internals.append({"type": "bond", "idx": (i, j)})

    for j in range(nat):
        for i in adjacency[j]:
            for k in adjacency[j]:
                if i < k:
                    internals.append({"type": "angle", "idx": (i, j, k)})

    for i, j in bonds:
        for k in adjacency[j]:
            for l in adjacency[k]:
                if i < l and k != i and l != j:
                    internals.append({"type": "dihedral", "idx": (i, j, k, l)})

    return internals, np.array(internals_to_values(internals, coords))


def internals_to_values(internals, coords):
    vals = []
    for it in internals:
        idx = it["idx"]
        if it["type"] == "bond":
            vals.append(dist(coords[idx[0]], coords[idx[1]]))
        elif it["type"] == "angle":
            vals.append(angle(coords[idx[0]], coords[idx[1]], coords[idx[2]]))
        else:
            vals.append(dihedral(coords[idx[0]], coords[idx[1]], coords[idx[2]], coords[idx[3]]))
    return vals


# ---------------------
# Internal → Cartesian
# ---------------------
def internal_to_cartesian(cart0, internals, target):
    nat = cart0.shape[0]

    def residuals(x):
        return np.array(internals_to_values(internals, x.reshape(nat, 3))) - target

    res = least_squares(
        residuals,
        cart0.reshape(-1),
        ftol=CONFIG["LSQRTOL"],
        xtol=CONFIG["LSQRTOL"],
        gtol=CONFIG["LSQRTOL"],
        max_nfev=CONFIG["LSQRMAXITER"],
    )
    return res.x.reshape(nat, 3)


# ---------------------
# CBS energy
# ---------------------
def cbs_compose(c1, c2, s1, s2):
    A = (2 ** 3) / (3 ** 3 - 2 ** 3)
    corr = c2 + A * (c2 - c1)
    denom = math.exp(CONFIG["BETA_HF"] * 3) - math.exp(CONFIG["BETA_HF"] * 2)
    hf = s2 + (math.exp(CONFIG["BETA_HF"] * 3) * (s2 - s1)) / denom
    return hf + corr, hf, corr


def compute_scf_and_correlation(symbols, coords, basis):
    mol = gto.M(atom=list(zip(symbols, coords)), basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    try:
        mycc = cc.CCSD(mf).run()
        corr = mycc.e_corr + (mycc.ccsd_t() or 0.0)
    except Exception:
        corr = mp.MP2(mf).run().e_corr
    return mf.e_tot, corr


def compute_cbs_energy(symbols, coords, basis_pair):
    s1, c1 = compute_scf_and_correlation(symbols, coords, basis_pair[0])
    s2, c2 = compute_scf_and_correlation(symbols, coords, basis_pair[1])
    return cbs_compose(c1, c2, s1, s2)


# ---------------------
# Optimization
# ---------------------
def optimize_geometry(symbols, coords0, basis_pair):
    internals, x0 = build_internals(symbols, coords0)

    opt_history = []
    cycle = {"i": 0}

    def objective(x):
        nonlocal opt_history
        cart = internal_to_cartesian(coords0, internals, x)
        E, Ehf, Ec = compute_cbs_energy(symbols, cart, basis_pair)

        cycle["i"] += 1
        opt_history.append({
            "cycle": int(cycle["i"]),
            "cbs_energy": float(E),
        })

        print(f"opt step: E_cbs = {E:.10f} Ha")
        return E

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"ftol": CONFIG["ENERGY_TOL"], "maxiter": CONFIG["LBFGSB_MAXITER"]},
    )

    cart_final = internal_to_cartesian(coords0, internals, res.x)
    E_final, Ehf, Ec = compute_cbs_energy(symbols, cart_final, basis_pair)

    return {
        "history": opt_history,
        "final_energy": E_final,
        "final_hf_cbs": Ehf,
        "final_corr_cbs": Ec,
        "final_cart": cart_final,
        "x_opt": res.x,
        "opt_result": res,
    }


# ---------------------
# CLI
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_xyz")
    parser.add_argument("-o", "--output", default="opt_out.xyz")
    args = parser.parse_args()

    symbols, coords = read_xyz(args.input_xyz)
    res = optimize_geometry(symbols, coords, CONFIG["BASIS_SETS"])

    write_xyz(args.output, symbols, res["final_cart"],
              f"CBS opt energy {res['final_energy']:.10f} Ha")

    print("Optimization finished.")
    print("Final CBS energy:", res["final_energy"])


if __name__ == "__main__":
    main()
