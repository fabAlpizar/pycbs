import argparse
import math
import sys
from configparser import SectionProxy
import os

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
    "X1_CORR": 2.0,
    "X2_CORR": 3.0,
    # HF extrapolation model uses an exponential; BETA often ~ 1.60
    "X1_HF": 2.0,
    "X2_HF": 3.0,
    "BETA_HF": 1.63,

    # PySCF settings
    "PYSCF_MAX_MEMORY": 4 * 1024,  # in MB
    "PYSCF_NTHREADS": 1,

    # Geometry inversion (least_squares) options
    "LSQRTOL": 1e-8,
    "LSQRMAXITER": 200,

    # Optimization (L-BFGS-B) options
    "ENERGY_TOL": 1e-6,    # Hartree stopping criterion for CBS energy change
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
    # normalize preference to lower-case tokens
    pref = tuple([p.lower() for p in method_preference]) if method_preference is not None else ("ccsd(t)", "mp2")

    if "ccsd(t)" in pref:
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

    if corr_e is None and ("mp2" in pref or "mp2" in method_preference):
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
def compute_cbs_energy(symbols, coords, basis_pair=None, cfg=CONFIG, method_preference=None):
    """
    Compute CBS-extrapolated energy for given Cartesian coordinates.
    Returns: (E_CBS, E_HF_CBS, E_CORR_CBS, debug_dict)

    method_preference: tuple/list of method tokens passed down to compute_scf_and_correlation
                       e.g. ("ccsd(t)", "mp2") or ("mp2",)
    """
    if basis_pair is None:
        basis_pair = cfg["BASIS_SETS"]
    # small basis first
    bs1, bs2 = basis_pair

    scf1, corr1 = compute_scf_and_correlation(symbols, coords, bs1, method_preference=method_preference)
    scf2, corr2 = compute_scf_and_correlation(symbols, coords, bs2, method_preference=method_preference)

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
    def __init__(self, symbols, internals, coords0, rounding=8, method_preference=None):
        self.symbols = symbols
        self.internals = internals
        self.coords0 = np.array(coords0)
        self.rounding = rounding
        self.method_preference = method_preference
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
        # compute CBS, passing method preference if present
        E_cbs, E_hf_cbs, E_corr_cbs, debug = compute_cbs_energy(
            self.symbols, cart, basis_pair, cfg=CONFIG, method_preference=self.method_preference
        )
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
    # record per-cycle CBS energies
    cbs_history = []

    if options is None:
        options = {}
    cfg = CONFIG

    # Map options.method (string like 'MP2' or 'CCSD(T)') into method_preference tuple
    method_pref = None
    method_opt = options.get("method", None)
    if method_opt is not None:
        mstr = str(method_opt).strip().lower()
        if "mp2" in mstr and "ccsd" not in mstr:
            method_pref = ("mp2",)
        elif "ccsd" in mstr or "ccsd(t)" in mstr or "ccsdt" in mstr:
            # Prefer CCSD(T) and fall back to MP2 if needed
            method_pref = ("ccsd(t)", "mp2")
        else:
            # unknown token: keep default behavior
            method_pref = ("ccsd(t)", "mp2")
    else:
        method_pref = ("ccsd(t)", "mp2")

    internals, values0 = build_internals(symbols, coords0)
    energy_cache = EnergyCache(symbols, internals, coords0, rounding=8, method_preference=method_pref)

    # bounds: for bonds only, apply a factor to initial distances for bounds
    nvars = len(internals)
    bounds = [(None, None)] * nvars
    for idx, it in enumerate(internals):
        if it["type"] == "bond":
            i, j = it["idx"]
            r0 = values0[idx]
            bounds[idx] = (cfg["BOND_MIN_FACTOR"] * r0, cfg["BOND_MAX_FACTOR"] * r0)

    # objective function for scipy minimize: returns scalar CBS energy
    def objective(x):
        res = energy_cache.evaluate(x, basis_pair=basis_pair)
        return res["E_cbs"]

    # provide a simple callback to mirror progress (not too verbose)
    last = {"E": None}
    def callback(xk):
        E = energy_cache.evaluate(xk, basis_pair=basis_pair)["E_cbs"]
        if last["E"] is None or abs(E - last["E"]) > 1e-8:
            print(f"opt step: E_cbs = {E:.10f} Ha")
            last["E"] = E
        # record CBS energy for this optimization cycle (append in Ha)
        cbs_history.append(E)

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
    # ensure final energy is in history (avoid duplicate if already appended)
    final_E = final["E_cbs"]
    if not cbs_history or abs(cbs_history[-1] - final_E) > 1e-12:
        cbs_history.append(final_E)

    result = {
        "opt_result": opt_res,
        "final_energy": final["E_cbs"],
        "final_hf_cbs": final["E_hf_cbs"],
        "final_corr_cbs": final["E_corr_cbs"],
        "final_cart": final_cart,
        "debug": final["debug"],
        "internals": internals,
        "x_opt": x_opt,
        "cbs_history": cbs_history,
    }
    return result


# ---------------------
# Helper: clean inline comment-bearing config values
# ---------------------
def _clean_config_value(raw):
    """
    Given a raw string returned by configparser.SectionProxy.get, remove inline comments
    like '; comment' or '# comment' and strip whitespace. Return None if input falsy.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    # remove inline comments starting with ';' or '#'
    val = raw.split(';', 1)[0].split('#', 1)[0].strip()
    return val if val != "" else None


# ---------------------
# Programmatic wrapper for integration with cli.py
# ---------------------
def optimize_from_config(cfg_section: SectionProxy):
    """
    Read an [optimization] config section and run the optimizer.
    Returns a list of dicts: [{"cycle": int, "cbs_energy": float}, ...]
    """
    # Parse enable flag (accepts bool-like strings) with inline-comment robustness
    raw_enabled = None
    try:
        # try the SectionProxy convenience first
        raw_enabled = cfg_section.get("optimization")
    except Exception:
        raw_enabled = cfg_section.get("optimization", None) if isinstance(cfg_section, dict) else None

    enabled_val = _clean_config_value(raw_enabled)
    if enabled_val is None:
        enabled = False
    else:
        enabled = str(enabled_val).lower() in ("1", "true", "yes", "on")

    if not enabled:
        return []

    # xyz is mandatory
    raw_xyz = cfg_section.get("xyz", fallback=None)
    xyz = _clean_config_value(raw_xyz)
    if not xyz:
        raise ValueError("INI [optimization] must include 'xyz' when optimization = True")

    if not os.path.exists(xyz):
        raise FileNotFoundError(f"XYZ file not found: {xyz}")

    # read xyz (uses existing read_xyz in this module)
    symbols, coords0 = read_xyz(xyz)

    # method / basis pair
    raw_method = cfg_section.get("method", fallback=None)
    method = _clean_config_value(raw_method) or "CCSD(T)"
    raw_basis1 = cfg_section.get("basis1", fallback=None)
    raw_basis2 = cfg_section.get("basis2", fallback=None)
    basis1 = _clean_config_value(raw_basis1)
    basis2 = _clean_config_value(raw_basis2)
    basis_pair = None
    if basis1 and basis2:
        basis_pair = (basis1.strip(), basis2.strip())
    else:
        basis_pair = CONFIG["BASIS_SETS"]

    # numeric options
    raw_beta = cfg_section.get("beta", fallback=None)
    beta_val = _clean_config_value(raw_beta)
    try:
        beta = float(beta_val) if beta_val is not None else None
    except Exception:
        beta = None

    raw_spin = cfg_section.get("spin", fallback=None)
    spin_val = _clean_config_value(raw_spin)
    try:
        spin = int(spin_val) if spin_val is not None else None
    except Exception:
        spin = None

    # output file for the final optimized xyz (optional)
    raw_output = cfg_section.get("output", fallback=None)
    output_xyz = _clean_config_value(raw_output)
    if output_xyz is None:
        base, ext = os.path.splitext(xyz)
        output_xyz = base + "_opt" + (ext or ".xyz")

    # Build options dict for optimize_geometry
    options = {}
    if beta is not None:
        options["beta"] = beta
    if spin is not None:
        options["spin"] = spin
    options["method"] = method

    # Run the core optimizer (returns dict with cbs_history)
    res = optimize_geometry(symbols, coords0, basis_pair=basis_pair, options=options)

    # Write final optimized geometry if available (non-fatal if it fails)
    # --- WRITE OPTIMIZED GEOMETRY (MANDATORY OUTPUT) ---
    opt_coords = res.get("final_cart", None)
    if opt_coords is None:
        raise RuntimeError(
            "Optimization completed but no final_cart returned; cannot write optimized geometry."
        )

    output_xyz = os.path.abspath(output_xyz)
    with open(output_xyz, "w", encoding="utf-8") as fh:
        fh.write(f"{len(symbols)}\n")
        fh.write(
            f"Optimized geometry (CBS). Final E = {res['final_energy']:.10f} Ha\n"
        )
        for s, xyz in zip(symbols, opt_coords):
            fh.write(f"{s} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")

    print(f"[optimization] Optimized geometry written to: {output_xyz}")

    # Build opt_cycles list from cbs_history
    cbs_history = res.get("cbs_history", [])
    opt_cycles = []
    for i, e_cbs in enumerate(cbs_history, start=1):
        opt_cycles.append({"cycle": i, "cbs_energy": e_cbs})

    return opt_cycles


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
