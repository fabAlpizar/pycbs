#!/usr/bin/env python3
"""
L-BFGS-B CBS optimizer (internals + L-BFGS) with full parameter parity with the
ALL-PER-CYCLE variant: accepts 'method', 'maxcycle', 'fac', 'cut', 'workers',
'energy_accept_tol', 'energy_crit', 'frozen', etc.

Behavior:
 - Optimization strategy: L-BFGS-B on redundant internal coordinates (3-point helper
   retained for diagnostics).
 - Correlation methods: CCSD(T) preferred if method startswith "CCSD", otherwise MP2.
 - Frozen core: supports integer, explicit list, or 'set_frozen' / 'auto' token.
 - workers: used to evaluate small/large basis concurrently.
"""
from pathlib import Path
import argparse
import math
import sys
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# PySCF imports
try:
    from pyscf import gto, scf, mp, cc, lib
except Exception as e:
    raise ImportError("PySCF is required for this script. Install pyscf and retry.") from e

from pycbs.writer import write_cycle_energies, write_final_xyz

# ---------------------
# CONFIGURATION (defaults)
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
    "PYSCF_NTHREADS": 6,
    "PYSCF_VERBOSE": 1,            # 0 = quiet, >0 prints PySCF output during calls

    # Geometry inversion (least_squares) options
    "LSQRTOL": 1e-8,
    "LSQRMAXITER": 200,

    # Optimization (L-BFGS-B) options
    "ENERGY_TOL": 1e-8,  # Hartree stopping criterion for CBS energy change (ftol)
    "X_TOL": 1e-4,  # step tolerance for internal coordinates
    "LBFGSB_MAXITER": 200,
    "BOND_MIN_FACTOR": 0.5,  # lower bound for bond length relative to initial
    "BOND_MAX_FACTOR": 2.0,  # upper bound for bond length relative to initial
}

# module-level default spin (can be set via _update_config_from_params)
DEFAULT_SPIN = 0

# reflect config threads
lib.num_threads(CONFIG["PYSCF_NTHREADS"])

# ---------------------
# Frozen parsing helpers (normalize to hashable token)
# ---------------------
def normalize_frozen_param(frozen_raw) -> Optional[Tuple]:
    """
    Normalize `frozen` param into a hashable tuple token for caching.
    Accepts None, int, [ints], or strings like "2", "0,1", "[0,1]", "set_frozen", "auto".
    Returns: None | ('set_frozen',) | ('int', n) | ('list', i0,i1,...)
    """
    if frozen_raw is None:
        return None
    if isinstance(frozen_raw, int):
        return ('int', int(frozen_raw))
    if isinstance(frozen_raw, (list, tuple)):
        try:
            ints = tuple(int(x) for x in frozen_raw)
        except Exception:
            raise ValueError("If passing a list for 'frozen' it must contain integer indices.")
        if len(ints) == 0:
            return None
        return ('list',) + ints
    if isinstance(frozen_raw, str):
        s = frozen_raw.strip()
        low = s.lower()
        if low in ('set_frozen', 'setfrozen', 'auto', 'autodetect'):
            return ('set_frozen',)
        tokens = re.findall(r'-?\d+', s)
        if tokens:
            ints = tuple(int(t) for t in tokens)
            if len(ints) == 1:
                return ('int', ints[0])
            return ('list',) + ints
        try:
            n = int(s)
            return ('int', n)
        except Exception:
            raise ValueError(f"Unrecognized frozen value: {frozen_raw!r}. Expected integer, list, or 'set_frozen'.")
    raise ValueError(f"Unsupported frozen param type: {type(frozen_raw)}")


def frozen_token_to_pyscf_arg(frozen_token):
    """Convert normalized token into PySCF arg or 'set_frozen' marker."""
    if frozen_token is None:
        return None
    if frozen_token[0] == 'set_frozen':
        return 'set_frozen'
    if frozen_token[0] == 'int':
        return int(frozen_token[1])
    if frozen_token[0] == 'list':
        return [int(x) for x in frozen_token[1:]]
    return None

# ---------------------
# Geometry helpers (unchanged)
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

    from scipy.optimize import least_squares
    result = least_squares(residuals, x0, method="trf", ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)
    if not result.success:
        print("Warning: internal->cartesian least_squares did not converge: " + result.message, file=sys.stderr)
    return result.x.reshape((nat, 3))


# ---------------------
# 3-point parabolic helper (robust)
# ---------------------
def parabolic_minimum_3pt(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != 3 or y.size != 3:
        idx = np.argmin(y)
        return float(x[idx]), float(y[idx])
    try:
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        if abs(a) < 1e-20:
            idx = np.argmin(y)
            return float(x[idx]), float(y[idx])
        x_min = -b / (2.0 * a)
        y_min = a * x_min ** 2 + b * x_min + c
        return float(x_min), float(y_min)
    except Exception:
        idx = np.argmin(y)
        return float(x[idx]), float(y[idx])


# ---------------------
# CBS composition
# ---------------------
def cbs_compose(corr_small, corr_big, scf_small, scf_big, cfg=CONFIG):
    X1 = cfg["X1_CORR"]
    X2 = cfg["X2_CORR"]
    X1_HF = cfg["X1_HF"]
    X2_HF = cfg["X2_HF"]
    BETA = cfg["BETA_HF"]

    a_corr = (X1 ** 3) / (X2 ** 3 - X1 ** 3)
    E_corr_cbs = corr_big + a_corr * (corr_big - corr_small)

    denom = math.exp(BETA * X2_HF) - math.exp(BETA * X1_HF)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    b_hf = math.exp(BETA * X1_HF) / denom

    E_hf_cbs = scf_big + b_hf * (scf_big - scf_small)

    E_cbs = E_hf_cbs + E_corr_cbs
    return float(E_cbs), float(E_hf_cbs), float(E_corr_cbs)


# ---------------------
# Energy evaluation with PySCF (supports frozen_token and method preference)
# ---------------------
def compute_scf_and_correlation(symbols, coords, basis, frozen_token=None, method_preference=("ccsd(t)", "mp2")):
    """
    Evaluate SCF + correlation for a given basis. frozen_token normalized token.
    method_preference is a tuple of method names in order of preference (strings).
    """
    nat = len(symbols)
    mol = gto.Mole()
    mol.verbose = CONFIG["PYSCF_VERBOSE"]
    mol.unit = "Angstrom"
    mol.atom = [(symbols[i], tuple(coords[i].tolist())) for i in range(nat)]
    mol.basis = basis
    mol.charge = 0
    mol.spin = int(DEFAULT_SPIN)
    mol.build()

    mol.max_memory = CONFIG["PYSCF_MAX_MEMORY"]
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-8
    mf.verbose = CONFIG["PYSCF_VERBOSE"]
    mf.max_cycle = 200
    try:
        scf_e = mf.kernel()
    except Exception as e:
        raise RuntimeError("SCF failed: " + str(e))

    corr_e = None
    last_err = None
    pyscf_frozen_arg = frozen_token_to_pyscf_arg(frozen_token)

    # CCSD(T) branch first if requested
    if "ccsd(t)" in [m.lower() for m in method_preference]:
        try:
            if pyscf_frozen_arg == 'set_frozen':
                mycc = cc.CCSD(mf)
                mycc.set_frozen()
            else:
                mycc = cc.CCSD(mf, frozen=pyscf_frozen_arg) if pyscf_frozen_arg is not None else cc.CCSD(mf)
            mycc.conv_tol = 1e-7
            mycc.max_cycle = 200
            mycc.verbose = CONFIG["PYSCF_VERBOSE"]
            mycc.kernel()
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

    # MP2 fallback / preference
    if corr_e is None and "mp2" in [m.lower() for m in method_preference]:
        try:
            if pyscf_frozen_arg == 'set_frozen':
                mymp = mp.MP2(mf)
                mymp.set_frozen()
                res = mymp.run()
            else:
                mymp = mp.MP2(mf, frozen=pyscf_frozen_arg) if pyscf_frozen_arg is not None else mp.MP2(mf)
                mymp.verbose = CONFIG["PYSCF_VERBOSE"]
                res = mymp.run()
            mp2_corr = None
            if hasattr(res, "e_corr"):
                mp2_corr = float(res.e_corr)
            elif hasattr(mymp, "e_corr"):
                mp2_corr = float(mymp.e_corr)
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
        raise RuntimeError(f"No correlation energy available for basis {basis} (last error: {last_err})")

    return float(scf_e), float(corr_e)


# ---------------------
# High-level CBS energy: can run bs1/bs2 concurrently with workers
# ---------------------
def compute_cbs_energy(symbols, coords, basis_pair=None, cfg=CONFIG, frozen_token=None, method_preference=("ccsd(t)", "mp2"), workers: int = 1):
    if basis_pair is None:
        basis_pair = cfg["BASIS_SETS"]
    bs1, bs2 = basis_pair

    if workers is not None and int(workers) > 1:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut1 = ex.submit(compute_scf_and_correlation, symbols, coords, bs1, frozen_token, method_preference)
            fut2 = ex.submit(compute_scf_and_correlation, symbols, coords, bs2, frozen_token, method_preference)
            scf1, corr1 = fut1.result()
            scf2, corr2 = fut2.result()
    else:
        scf1, corr1 = compute_scf_and_correlation(symbols, coords, bs1, frozen_token, method_preference)
        scf2, corr2 = compute_scf_and_correlation(symbols, coords, bs2, frozen_token, method_preference)

    E_cbs, E_hf_cbs, E_corr_cbs = cbs_compose(corr1, corr2, scf1, scf2, cfg=cfg)
    debug = {"scf1": scf1, "scf2": scf2, "corr1": corr1, "corr2": corr2, "E_hf_cbs": E_hf_cbs, "E_corr_cbs": E_corr_cbs}
    return E_cbs, E_hf_cbs, E_corr_cbs, debug


# ---------------------
# Caching wrapper (keyed by internals + frozen_token + method)
# ---------------------
class EnergyCache:
    def __init__(self, symbols, internals, coords0, rounding=8):
        self.symbols = symbols
        self.internals = internals
        self.coords0 = np.array(coords0)
        self.rounding = rounding
        self._cache = {}

    def _key_from_internals(self, internal_vector, frozen_token, method_name, basis_pair):
        base_key = tuple([round(float(x), self.rounding) for x in internal_vector])
        return (base_key, frozen_token, str(method_name), tuple(basis_pair))

    def evaluate(self, internal_vector, basis_pair=None, frozen_token=None, method_name="CCSD(T)", workers: int = 1):
        k = self._key_from_internals(internal_vector, frozen_token, method_name, basis_pair or CONFIG["BASIS_SETS"])
        if k in self._cache:
            return self._cache[k]
        cart = internal_to_cartesian(self.coords0, self.internals, internal_vector)
        method_pref = ("ccsd(t)", "mp2") if str(method_name).upper().startswith("CCSD") else ("mp2",)
        E_cbs, E_hf_cbs, E_corr_cbs, debug = compute_cbs_energy(self.symbols, cart, basis_pair, cfg=CONFIG,
                                                                 frozen_token=frozen_token,
                                                                 method_preference=method_pref,
                                                                 workers=workers)
        res = {"E_cbs": E_cbs, "E_hf_cbs": E_hf_cbs, "E_corr_cbs": E_corr_cbs, "cart": cart, "debug": debug}
        self._cache[k] = res
        return res


# ---------------------
# Optimization driver (L-BFGS-B) that accepts the same params as previous script
# ---------------------
def optimize_geometry(symbols, coords0, basis_pair=None, options=None, frozen_token=None, method_name="CCSD(T)", workers: int = 1, energy_accept_tol: Optional[float] = None, debug: bool = False):
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
        try:
            res = energy_cache.evaluate(xk, basis_pair=basis_pair, frozen_token=frozen_token, method_name=method_name, workers=workers)
            E = res["E_cbs"]
            last["count"] += 1
            dbg = res.get("debug", {})
            scf1 = dbg.get("scf1"); scf2 = dbg.get("scf2")
            corr1 = dbg.get("corr1"); corr2 = dbg.get("corr2")
            E_hf_cbs = dbg.get("E_hf_cbs"); E_corr_cbs = dbg.get("E_corr_cbs")
            print(f"[opt step {last['count']}] E_cbs = {E:.10f} Ha  (E_hf_cbs={E_hf_cbs:.10f}, E_corr_cbs={E_corr_cbs:.10f})")
            if scf1 is not None:
                print(f"    scf (small) = {scf1:.10f}, scf (big) = {scf2:.10f}")
            if corr1 is not None:
                print(f"    corr (small) = {corr1:.10f}, corr (big) = {corr2:.10f}")
            history.append({'cycle': last["count"], 'energy': float(E)})
            xi = np.array(xk, dtype=float)
            print(f"    internals (n={len(xi)}): min={xi.min():.6e}, max={xi.max():.6e}, rms={np.sqrt(np.mean(xi**2)):.6e}")
            rms_cart = np.sqrt(np.mean((res["cart"].reshape(-1) - np.array(coords0).reshape(-1))**2))
            print(f"    cart RMS from initial: {rms_cart:.6e} Å")
            if energy_accept_tol is not None and last["E"] is not None:
                # extra sanity check (not native to L-BFGS-B): print if last step improved less than threshold
                ediff = last["E"] - E
                print(f"    ΔE (prev - current) = {ediff:.4e} Ha (energy_accept_tol={energy_accept_tol})")
            last["E"] = E
        except Exception as e:
            print(f"[callback] evaluation failed: {e}", file=sys.stderr)

    x0 = np.array(values0, dtype=float)
    # initial energy entry
    try:
        E0 = energy_cache.evaluate(x0, basis_pair=basis_pair, frozen_token=frozen_token, method_name=method_name, workers=workers)["E_cbs"]
        history.append({'cycle': 0, 'energy': float(E0)})
        print(f"[init] E_cbs (initial) = {E0:.10f} Ha")
    except Exception as e:
        print("[init] initial energy eval failed:", e, file=sys.stderr)

    opt_res = minimize(
        lambda x: energy_cache.evaluate(x, basis_pair=basis_pair, frozen_token=frozen_token, method_name=method_name, workers=workers)["E_cbs"],
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
    final = energy_cache.evaluate(x_opt, basis_pair=basis_pair, frozen_token=frozen_token, method_name=method_name, workers=workers)
    final_cart = final["cart"]
    result = {
        "opt_result": opt_res,
        "final_energy": final["E_cbs"],
        "final_hf_cbs": final["E_hf_cbs"],
        "final_corr_cbs": final["E_corr_cbs"],
        "final_cart": final_cart,
        "debug": final.get("debug", {}),
        "internals": internals,
        "x_opt": x_opt,
        "history": history,
    }
    return result


# ---------------------
# Helper to update CONFIG from params (now supports all params requested)
# ---------------------
def _update_config_from_params(params):
    global DEFAULT_SPIN
    if params is None:
        return
    try:
        if "X1" in params and params["X1"] is not None:
            CONFIG["X1_CORR"] = float(params["X1"])
        if "X2" in params and params["X2"] is not None:
            CONFIG["X2_CORR"] = float(params["X2"])
        if "X1hf" in params and params["X1hf"] is not None:
            CONFIG["X1_HF"] = float(params["X1hf"])
        if "X2hf" in params and params["X2hf"] is not None:
            CONFIG["X2_HF"] = float(params["X2hf"])
        if "beta" in params and params["beta"] is not None:
            CONFIG["BETA_HF"] = float(params["beta"])
        if "pyscf_threads" in params and params["pyscf_threads"] is not None:
            nt = int(params["pyscf_threads"])
            CONFIG["PYSCF_NTHREADS"] = nt
            lib.num_threads(nt)
        if "pyscf_max_memory" in params and params["pyscf_max_memory"] is not None:
            CONFIG["PYSCF_MAX_MEMORY"] = int(params["pyscf_max_memory"])
        if "pyscf_verbose" in params and params["pyscf_verbose"] is not None:
            CONFIG["PYSCF_VERBOSE"] = int(params["pyscf_verbose"])
        if params.get("basis1") and params.get("basis2"):
            CONFIG["BASIS_SETS"] = (params["basis1"], params["basis2"])

        # map maxcycle to LBFGSB_MAXITER
        if "maxcycle" in params and params["maxcycle"] is not None:
            CONFIG["LBFGSB_MAXITER"] = int(params["maxcycle"])
        # map energy_crit and energy_accept_tol to ENERGY_TOL if provided (user controls both)
        if "energy_crit" in params and params["energy_crit"] is not None:
            try:
                CONFIG["ENERGY_TOL"] = float(params["energy_crit"])
            except Exception:
                pass
        if "energy_accept_tol" in params and params["energy_accept_tol"] is not None:
            try:
                # keep as separate param as well; it's used as extra check in callback
                params["_energy_accept_tol_internal"] = float(params["energy_accept_tol"])
            except Exception:
                pass
        # keep fac / cut in params for API parity (not used by L-BFGS internals)
        # spin override
        if "spin" in params and params["spin"] is not None:
            DEFAULT_SPIN = int(params["spin"])
    except Exception:
        pass


# ---------------------
# Programmatic API (accepts full param set)
# ---------------------
def run_optimization(params: dict, outputs_dir: Path):
    """
    params: dict expected keys (a superset of):
        input_xyz, method, X1, X2, X1hf, X2hf, beta, maxcycle, fac, spin,
        debug, energy_accept_tol, workers, energy_crit, cut, frozen, basis1, basis2, pyscf_threads,...
    outputs_dir: Path where outputs must be stored
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    _update_config_from_params(params)

    input_xyz = params.get("input_xyz") or params.get("input") or params.get("geometry")
    if not input_xyz:
        raise ValueError("params must include 'input_xyz' (path to input XYZ)")
    symbols, coords0 = read_xyz(input_xyz)

    # normalize frozen param
    frozen_raw = params.get("frozen", None)
    try:
        frozen_token = normalize_frozen_param(frozen_raw)
    except Exception as e:
        raise ValueError(f"Invalid frozen parameter: {e}")

    # method selection
    method = params.get("method", "CCSD(T)")
    method_name = str(method)

    # workers for concurrent basis evaluation
    workers = int(params.get("workers", 1) or 1)

    # energy_accept_tol (optional extra check used in callback)
    energy_accept_tol = params.get("energy_accept_tol", None)
    if energy_accept_tol is not None:
        try:
            energy_accept_tol = float(energy_accept_tol)
        except Exception:
            energy_accept_tol = None

    # debug flag
    debug = bool(params.get("debug", False))

    # basis pair override
    if params.get("basis1") and params.get("basis2"):
        basis_pair = (params["basis1"], params["basis2"])
    else:
        basis_pair = CONFIG["BASIS_SETS"]

    print("Starting L-BFGS-B optimization with basis pair:", basis_pair)
    print(f"PySCF verbosity = {CONFIG['PYSCF_VERBOSE']}, threads = {CONFIG['PYSCF_NTHREADS']}, max mem = {CONFIG['PYSCF_MAX_MEMORY']} MB")
    if frozen_token is None:
        print("Frozen core: none (all electrons correlated)")
    else:
        print("Frozen core token:", frozen_token)
    print("Method:", method_name, "workers:", workers, "debug:", debug)

    res = optimize_geometry(symbols, coords0, basis_pair=basis_pair, frozen_token=frozen_token, method_name=method_name, workers=workers, energy_accept_tol=energy_accept_tol, debug=debug)

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
# CLI entrypoint exposing same args as previous script (compat)
# ---------------------
def _cli_main():
    parser = argparse.ArgumentParser(description="L-BFGS-B CBS geometry optimizer (internals + L-BFGS).")
    parser.add_argument("-i", "--input", required=True, help="Input XYZ file")
    parser.add_argument("-o", "--out", default="PyCBS-OUTPUTS", help="Outputs directory")
    parser.add_argument("--method", default="CCSD(T)")
    parser.add_argument("--maxcycle", type=int, default=CONFIG["LBFGSB_MAXITER"])
    parser.add_argument("--fac", type=float, default=0.05, help="(API parity; not used by L-BFGS internals)")
    parser.add_argument("--x1", type=float, default=None)
    parser.add_argument("--x2", type=float, default=None)
    parser.add_argument("--x1hf", type=float, default=None)
    parser.add_argument("--x2hf", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Enable verbose diagnostics")
    parser.add_argument("--energy_accept_tol", type=float, default=None, help="Per-move acceptance energy (Ha). Used as optional diagnostic check.")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for concurrent basis evaluations (default 1)")
    parser.add_argument("--energy_crit", type=float, default=CONFIG["ENERGY_TOL"], help="Cycle convergence energy (Ha). Mapped to ftol for L-BFGS-B.")
    parser.add_argument("--cut", type=float, default=0.75, help="(API parity; not used by L-BFGS internals)")
    parser.add_argument("--frozen", type=str, default=None, help="Frozen core option. Examples: '2', '[0,1]', '0,1,16', 'set_frozen' (auto).")
    parser.add_argument("--basis1", default=None, help="Smaller basis override")
    parser.add_argument("--basis2", default=None, help="Larger basis override")
    parser.add_argument("--pyscf_threads", type=int, default=None, help="Set PySCF thread count")
    parser.add_argument("--pyscf_max_memory", type=int, default=None, help="Set PySCF max memory (MB)")
    parser.add_argument("--pyscf_verbose", type=int, default=None, help="PySCF verbosity (0=quiet, >0 verbose)")
    args = parser.parse_args()

    params = {
        "input_xyz": args.input,
        "method": args.method,
        "X1": args.x1,
        "X2": args.x2,
        "X1hf": args.x1hf,
        "X2hf": args.x2hf,
        "beta": args.beta,
        "maxcycle": args.maxcycle,
        "fac": args.fac,
        "spin": args.spin,
        "debug": args.debug,
        "energy_accept_tol": args.energy_accept_tol,
        "workers": args.workers,
        "energy_crit": args.energy_crit,
        "cut": args.cut,
        "frozen": args.frozen,
        "basis1": args.basis1,
        "basis2": args.basis2,
        "pyscf_threads": args.pyscf_threads,
        "pyscf_max_memory": args.pyscf_max_memory,
        "pyscf_verbose": args.pyscf_verbose
    }
    outputs_dir = Path(args.out)
    result = run_optimization(params, outputs_dir)
    # also write the final single-file xyz from this CLI flag
    write_xyz(Path.cwd() / "optimized.xyz", result["symbols"], result["final_cart"],
              comment=f"CBS opt energy {result['final_energy']:.10f} Ha")
    print("Optimization finished. Final CBS energy (Ha):", result["final_energy"])
    print("Wrote optimized geometry to:", Path.cwd() / "optimized.xyz")
    print("Cycle energies and final XYZ written to:", outputs_dir)


if __name__ == "__main__":
    _cli_main()