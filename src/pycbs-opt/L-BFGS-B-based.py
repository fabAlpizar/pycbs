#!/usr/bin/env python3
"""
L-BFGS-B-based optimizer module (full implementation).

This module implements the L-BFGS-B optimization pathway using redundant
internal coordinates and CBS-extrapolated energies (PySCF). It exposes a
programmatic API:

    run_optimization(params: dict, outputs_dir: pathlib.Path) -> dict

which is the interface consumed by opt_cli.py.

The implementation is adapted from the opt_3.py optimizer implementation
and packaged here so that the optimizer can be run either directly from
the command line or called by the CLI dispatcher.

Notes:
- params is expected to contain keys normalized by opt_cli.prepare_options_from_params,
  e.g. 'input_xyz', 'method', 'X1', 'X2', 'X1hf', 'X2hf', 'beta', ...
- outputs_dir is a pathlib.Path pointing to the directory where outputs
  (cycle CSV and final XYZ) will be written. The function will create the folder
  if it does not exist.
"""
from pathlib import Path
import argparse
import math
import sys
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize, least_squares

# PySCF imports
try:
    from pyscf import gto, scf, mp, cc, lib
except Exception as e:
    raise ImportError("PySCF is required for this script. Install pyscf and retry.") from e

from pycbs.writer import write_cycle_energies, write_final_xyz

# ---------------------
# CONFIGURATION
# ---------------------
CONFIG = {
    # Basis sets: smaller first, larger second
    "BASIS_SETS": ("cc-pvdz", "cc-pvtz"),

    # CBS extrapolation parameters:
    "X1_CORR": 1.85,
    "X2_CORR": 2.639,
    "X1_HF": 3.02,
    "X2_HF": 3.64,
    "BETA_HF": 1.62,

    # PySCF settings
    "PYSCF_MAX_MEMORY": 4 * 1024,  # in MB
    "PYSCF_NTHREADS": 1,

    # Geometry inversion (least_squares) options
    "LSQRTOL": 1e-8,
    "LSQRMAXITER": 200,

    # Optimization (L-BFGS-B) options
    "ENERGY_TOL": 1e-8,  # Hartree stopping criterion for CBS energy change
    "X_TOL": 1e-4,  # step tolerance for internal coordinates
    "LBFGSB_MAXITER": 200,
    "BOND_MIN_FACTOR": 0.5,  # lower bound for bond length relative to initial
    "BOND_MAX_FACTOR": 2.0,  # upper bound for bond length relative to initial
}

# module-level default spin (can be set via _update_config_from_params)
DEFAULT_SPIN = 0


# ---------------------
# Utility geometry functions
# ---------------------
def read_xyz(filename):
    with open(filename) as fh:
        lines = [l.rstrip() for l in fh if l.strip()]
    natom = int(lines[0].split()[0])
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
    ba = a - b
    bc = c - b
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-16)
    cosv = np.clip(cosv, -1.0, 1.0)
    return math.acos(cosv)


def dihedral(a, b, c, d):
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

    angles = []
    adjacency = {i: set() for i in range(nat)}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for j in range(nat):
        for i in adjacency[j]:
            for k in adjacency[j]:
                if i < k:
                    angles.append((i, j, k))

    dihedrals = []
    for (i, j) in bonds:
        for k in adjacency[j]:
            if k == i:
                continue
            for l in adjacency[k]:
                if l == j:
                    continue
                if i < l:
                    dihedrals.append((i, j, k, l))

    internals = []
    for (i, j) in bonds:
        internals.append({"type": "bond", "idx": (i, j)})
    for (i, j, k) in angles:
        internals.append({"type": "angle", "idx": (i, j, k)})
    for (i, j, k, l) in dihedrals:
        internals.append({"type": "dihedral", "idx": (i, j, k, l)})

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
    nat = initial_cart.shape[0]
    x0 = initial_cart.reshape(-1)

    if lsq_opts is None:
        lsq_opts = {}
    tol = lsq_opts.get("tol", CONFIG["LSQRTOL"])
    max_nfev = lsq_opts.get("max_nfev", CONFIG["LSQRMAXITER"])

    def residuals(x):
        coords = x.reshape((nat, 3))
        vals = internals_to_values(internals, coords)
        diffs = []
        for it, v_calc, v_tgt in zip(internals, vals, target_values):
            if it["type"] == "bond":
                diffs.append(v_calc - v_tgt)
            elif it["type"] == "angle":
                d = v_calc - v_tgt
                d = (d + math.pi) % (2 * math.pi) - math.pi
                diffs.append(d)
            elif it["type"] == "dihedral":
                d = v_calc - v_tgt
                d = (d + math.pi) % (2 * math.pi) - math.pi
                diffs.append(d)
        return np.array(diffs)

    result = least_squares(residuals, x0, method="trf", ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)
    if not result.success:
        print("Warning: internal->cartesian least_squares did not converge: " + result.message, file=sys.stderr)
    return result.x.reshape((nat, 3))


# ---------------------
# CBS composition
# ---------------------
def cbs_compose(corr_small, corr_big, scf_small, scf_big, cfg=CONFIG):
    X1 = cfg["X1_CORR"]
    X2 = cfg["X2_CORR"]
    X1_HF = cfg["X1_HF"]
    X2_HF = cfg["X2_HF"]
    BETA = cfg["BETA_HF"]

    # correlation extrapolation coefficient using power -3
    a_corr = (X1 ** 3) / (X2 ** 3 - X1 ** 3)
    E_corr_cbs = corr_big + a_corr * (corr_big - corr_small)

    # HF prefactor b_hf as used in the SQM path:
    denom = math.exp(BETA * X2_HF) - math.exp(BETA * X1_HF)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    b_hf = math.exp(BETA * X1_HF) / denom

    # SQM-style HF composition (matches SQM module)
    E_hf_cbs = scf_big + b_hf * (scf_big - scf_small)

    E_cbs = E_hf_cbs + E_corr_cbs
    return float(E_cbs), float(E_hf_cbs), float(E_corr_cbs)


# ---------------------
# Energy evaluation with PySCF (returns scf_energy, corr_energy)
# ---------------------
def compute_scf_and_correlation(symbols, coords, basis, method_preference=("ccsd(t)", "mp2")):
    nat = len(symbols)
    mol = gto.Mole()
    mol.verbose = 0
    mol.unit = "Angstrom"
    mol.atom = [(symbols[i], tuple(coords[i].tolist())) for i in range(nat)]
    mol.basis = basis
    mol.charge = 0
    # Set spin from module-level default (can be changed by _update_config_from_params)
    mol.spin = int(DEFAULT_SPIN)
    mol.build()

    mol.max_memory = CONFIG["PYSCF_MAX_MEMORY"]
    mol.stdout = None

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-8
    mf.verbose = 0
    mf.max_cycle = 200
    try:
        scf_e = mf.kernel()
    except Exception as e:
        raise RuntimeError("SCF failed: " + str(e))

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
            ccsd_corr = getattr(mycc, "e_corr", None)
            if ccsd_corr is None:
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
    if basis_pair is None:
        basis_pair = cfg["BASIS_SETS"]
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
    def __init__(self, symbols, internals, coords0, rounding=8):
        self.symbols = symbols
        self.internals = internals
        self.coords0 = np.array(coords0)
        self.rounding = rounding
        self._cache = {}

    def _key_from_internals(self, internal_vector):
        return tuple([round(float(x), self.rounding) for x in internal_vector])

    def evaluate(self, internal_vector, basis_pair=None):
        k = self._key_from_internals(internal_vector)
        if k in self._cache:
            return self._cache[k]
        cart = internal_to_cartesian(self.coords0, self.internals, internal_vector)
        E_cbs, E_hf_cbs, E_corr_cbs, debug = compute_cbs_energy(self.symbols, cart, basis_pair)
        res = {"E_cbs": E_cbs, "E_hf_cbs": E_hf_cbs, "E_corr_cbs": E_corr_cbs, "cart": cart, "debug": debug}
        self._cache[k] = res
        return res


# ---------------------
# Optimization driver (collecting history)
# ---------------------
def optimize_geometry(symbols, coords0, basis_pair=None, options=None):
    if options is None:
        options = {}
    cfg = CONFIG

    internals, values0 = build_internals(symbols, coords0)
    energy_cache = EnergyCache(symbols, internals, coords0, rounding=8)

    nvars = len(internals)
    bounds = [(None, None)] * nvars
    for idx, it in enumerate(internals):
        if it["type"] == "bond":
            r0 = values0[idx]
            bounds[idx] = (cfg["BOND_MIN_FACTOR"] * r0, cfg["BOND_MAX_FACTOR"] * r0)
        elif it["type"] == "angle":
            bounds[idx] = (1e-2, math.pi - 1e-2)
        elif it["type"] == "dihedral":
            bounds[idx] = (-math.pi, math.pi)

    history = []
    last = {"E": None, "count": 0}

    def callback(xk):
        E = energy_cache.evaluate(xk, basis_pair=basis_pair)["E_cbs"]
        last["count"] += 1
        history.append({'cycle': last["count"], 'energy': float(E)})
        if last["E"] is None or abs(E - last["E"]) > 1e-8:
            print(f"opt step: E_cbs = {E:.10f} Ha")
            last["E"] = E

    x0 = np.array(values0, dtype=float)
    # initial energy entry
    try:
        E0 = energy_cache.evaluate(x0, basis_pair=basis_pair)["E_cbs"]
        history.append({'cycle': 0, 'energy': float(E0)})
    except Exception:
        pass

    opt_res = minimize(
        lambda x: energy_cache.evaluate(x, basis_pair=basis_pair)["E_cbs"],
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
        "history": history,
    }
    return result


# ---------------------
# Helper to update CONFIG from params
# ---------------------
def _update_config_from_params(params):
    # params keys: X1, X2, X1hf, X2hf, beta, basis1, basis2, pyscf_threads, pyscf_max_memory, spin
    global DEFAULT_SPIN
    if params is None:
        return
    try:
        if "X1" in params:
            CONFIG["X1_CORR"] = float(params["X1"])
        if "X2" in params:
            CONFIG["X2_CORR"] = float(params["X2"])
        if "X1hf" in params:
            CONFIG["X1_HF"] = float(params["X1hf"])
        if "X2hf" in params:
            CONFIG["X2_HF"] = float(params["X2hf"])
        if "beta" in params:
            CONFIG["BETA_HF"] = float(params["beta"])
        if "pyscf_threads" in params:
            nt = int(params["pyscf_threads"])
            CONFIG["PYSCF_NTHREADS"] = nt
            lib.num_threads(nt)
        if "pyscf_max_memory" in params:
            CONFIG["PYSCF_MAX_MEMORY"] = int(params["pyscf_max_memory"])
        # basis sets overrides
        if params.get("basis1") and params.get("basis2"):
            CONFIG["BASIS_SETS"] = (params["basis1"], params["basis2"])
        # spin override
        if "spin" in params:
            DEFAULT_SPIN = int(params["spin"])
    except Exception:
        # non-fatal; ignore invalid values (opt_cli should normalize)
        pass


# ---------------------
# Programmatic API
# ---------------------
def run_optimization(params: dict, outputs_dir: Path):
    """
    params: dict normalized by opt_cli.prepare_options_from_params(...)
    outputs_dir: Path where outputs must be stored (PyCBS-OUTPUTS)
    Returns: dict with keys 'history', 'final_energy', 'final_cart', 'symbols', 'outputs'
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    _update_config_from_params(params)

    input_xyz = params.get("input_xyz")
    if not input_xyz:
        raise ValueError("params must include 'input_xyz' (path to input XYZ)")
    symbols, coords0 = read_xyz(input_xyz)

    # determine basis pair (params can override)
    if params.get("basis1") and params.get("basis2"):
        basis_pair = (params["basis1"], params["basis2"])
    else:
        basis_pair = CONFIG["BASIS_SETS"]

    print("Starting L-BFGS-B optimization with basis pair:", basis_pair)
    res = optimize_geometry(symbols, coords0, basis_pair=basis_pair)

    final_cart = res["final_cart"]
    final_energy = float(res["final_energy"])
    history = res.get("history", [])

    base = Path(input_xyz).stem
    prefix = f"{base}_LBFGS"
    cycles_file = write_cycle_energies(outputs_dir, prefix, history)
    xyz_file = write_final_xyz(outputs_dir, prefix, symbols, final_cart, final_energy)

    return {"history": history, "final_energy": final_energy, "final_cart": final_cart, "symbols": symbols,
            "outputs": {"cycles": str(cycles_file), "xyz": str(xyz_file)}, "opt_result": res}


# ---------------------
# CLI entrypoint (backwards compatible)
# ---------------------
def _cli_main():
    parser = argparse.ArgumentParser(description="L-BFGS-B CBS geometry optimizer (internals + L-BFGS).")
    parser.add_argument("input_xyz", help="Input XYZ file")
    parser.add_argument("--output", "-o", default="opt_out.xyz", help="Output optimized XYZ")
    parser.add_argument("--basis1", default=None, help="Smaller basis set override")
    parser.add_argument("--basis2", default=None, help="Larger basis set override")
    parser.add_argument("--X1", type=float, default=None)
    parser.add_argument("--X2", type=float, default=None)
    parser.add_argument("--X1hf", type=float, default=None)
    parser.add_argument("--X2hf", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--spin", type=int, default=0)
    args = parser.parse_args()

    params = {
        "input_xyz": args.input_xyz,
        "basis1": args.basis1,
        "basis2": args.basis2,
        "X1": args.X1,
        "X2": args.X2,
        "X1hf": args.X1hf,
        "X2hf": args.X2hf,
        "beta": args.beta,
        "spin": args.spin,
    }
    outputs_dir = Path.cwd() / "PyCBS-OUTPUTS"
    result = run_optimization(params, outputs_dir)
    # also write the final single-file xyz from this CLI flag
    write_xyz(args.output, result["symbols"], result["final_cart"],
              comment=f"CBS opt energy {result['final_energy']:.10f} Ha")
    print("Optimization finished. Final CBS energy (Ha):", result["final_energy"])
    print("Wrote optimized geometry to:", args.output)
    print("Cycle energies and final XYZ written to:", outputs_dir)


if __name__ == "__main__":
    _cli_main()