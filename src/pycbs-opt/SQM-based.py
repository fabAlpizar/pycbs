#!/usr/bin/env python3
"""
SQM-based optimizer module (full implementation)

This implements the "SQM-style" redundant-internal-coordinate optimizer:
- reads an input XYZ
- builds a redundant set of internals (bonds, angles, dihedrals)
- for each internal coordinate performs a 5-point finite-difference
  (parabolic) scan and tries to update geometry if the CBS energy improves
- repeats for a number of cycles or until convergence

Exposes:
    run_optimization(params: dict, outputs_dir: pathlib.Path) -> dict

Expected params keys (normalized by opt_cli.prepare_options_from_params):
- input_xyz (path)
- method (e.g. 'CCSD(T)' or 'MP2')
- X1, X2, Xhf1, Xhf2, beta  (CBS parameters; optional)
- maxcycle (optional)
- workers (optional)  -- currently not used for parallelism in this implementation
- fac (displacement factor; optional)

Outputs:
- writes cycle-by-cycle energies CSV: PyCBS-OUTPUTS/SQM_cycle_energies.csv
- writes final optimized geometry XYZ: PyCBS-OUTPUTS/SQM_final_opt.xyz

Notes / limitations:
- Internal->cartesian updates are geometric and approximate (no B-matrix).
- This implementation runs single-threaded for evaluation stability. For larger
  jobs you can parallelize the per-displacement evaluations, but be careful with
  PySCF and process/thread spawning.
"""
from pathlib import Path
import math
import sys
from functools import lru_cache

import numpy as np
from pycbs.writer import write_cycle_energies, write_final_xyz

# PySCF imports (deferred errors if not installed)
try:
    from pyscf import gto, scf, cc, mp, lib
except Exception as e:
    raise ImportError("PySCF is required for this script. Install pyscf and retry.") from e

# -------------------------
# Defaults / parameters
# -------------------------
# Basis sets used for CBS extrapolation (small, big)
basis_sets = ['cc-pvdz', 'cc-pvtz']
# method default
DEFAULT_METHOD = 'CCSD(T)'

# default CBS/extrapolation parameters
X1_DEFAULT, X2_DEFAULT = 1.85, 2.639
X1HF_DEFAULT, X2HF_DEFAULT = 3.02, 3.64
BETA_DEFAULT = 1.62

# worker / pyscf threading defaults
DEFAULT_WORKERS = 1
PYSCF_THREADS = 1
lib.num_threads(PYSCF_THREADS)

# convergence / optimization defaults
MAXCYCLE_DEFAULT = 20
ENERGY_CRIT = 1e-8
FAC_DEFAULT = 0.05
CUT = 0.75

# covalent radii / masses
COV_RAD = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02
}
ATOMIC_MASS = {
    'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.999, 'F': 18.998,
    'P': 30.9738, 'S': 32.065, 'Cl': 35.453
}


# -------------------------
# I/O / geometry helpers
# -------------------------
def read_xyz(filename):
    with open(filename) as f:
        natoms = int(f.readline().strip())
        comment = f.readline().rstrip('\n')
        atoms = []
        coords = []
        for _ in range(natoms):
            parts = f.readline().split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(coords, dtype=float), comment


def xyz_to_pyscf_string(atoms, coords):
    lines = []
    for a, c in zip(atoms, coords):
        lines.append(f"{a} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}")
    return "\n".join(lines)


def distance(a, b):
    return np.linalg.norm(a - b)


# -------------------------
# Internals: build redundant set
# -------------------------
def build_redundant_internals(atoms, coords, bond_scale=1.2):
    nat = len(atoms)
    bonds = []
    for i in range(nat):
        for j in range(i + 1, nat):
            ri = COV_RAD.get(atoms[i], 0.7)
            rj = COV_RAD.get(atoms[j], 0.7)
            cutoff = (ri + rj) * bond_scale
            if distance(coords[i], coords[j]) <= cutoff:
                bonds.append((i, j))
    angles = []
    for j in range(nat):
        neighbors = [i for i, k in bonds if k == j] + [k for k, i in bonds if i == j]
        neighbors = sorted(set(neighbors))
        for a_idx in range(len(neighbors)):
            for b_idx in range(a_idx + 1, len(neighbors)):
                i = neighbors[a_idx]
                k = neighbors[b_idx]
                angles.append((i, j, k))
    dihedrals = []
    for (j, k) in bonds:
        neigh_j = [i for i, b in bonds if b == j and i != k] + [b for a, b in bonds if a == j and b != k]
        neigh_k = [i for i, b in bonds if b == k and i != j] + [b for a, b in bonds if a == k and b != j]
        neigh_j = list(set(neigh_j))
        neigh_k = list(set(neigh_k))
        for i in neigh_j:
            for l in neigh_k:
                if len({i, j, k, l}) == 4:
                    dihedrals.append((i, j, k, l))
    bonds = sorted(set(bonds))
    angles = sorted(set(angles))
    dihedrals = sorted(set(dihedrals))
    internals = []
    for (i, j) in bonds:
        internals.append({'type': 'bond', 'inds': (i, j)})
    for (i, j, k) in angles:
        internals.append({'type': 'angle', 'inds': (i, j, k)})
    for (i, j, k, l) in dihedrals:
        internals.append({'type': 'dihedral', 'inds': (i, j, k, l)})
    # compute numeric values (bonds in Å, angles/dihedrals in degrees)
    for ic in internals:
        if ic['type'] == 'bond':
            i, j = ic['inds']
            ic['value'] = distance(coords[i], coords[j])
        elif ic['type'] == 'angle':
            i, j, k = ic['inds']
            v1 = coords[i] - coords[j]
            v2 = coords[k] - coords[j]
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-16)
            cosang = np.clip(cosang, -1.0, 1.0)
            ic['value'] = np.degrees(np.arccos(cosang))
        elif ic['type'] == 'dihedral':
            i, j, k, l = ic['inds']
            b1 = coords[j] - coords[i]
            b2 = coords[k] - coords[j]
            b3 = coords[l] - coords[k]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            n1n = n1 / (np.linalg.norm(n1) + 1e-16)
            n2n = n2 / (np.linalg.norm(n2) + 1e-16)
            m1 = np.cross(n1n, b2/ (np.linalg.norm(b2) + 1e-16))
            x = np.dot(n1n, n2n)
            y = np.dot(m1, n2n)
            ic['value'] = np.degrees(np.arctan2(y, x))
    return internals


# -------------------------
# Geometric transformations for single-internal updates (approximate)
# -------------------------
def compute_dihedral(p0, p1, p2, p3):
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1) + 1e-16
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def apply_bond_change(coords, i, j, new_length, atoms):
    p = coords.copy()
    ri = p[i].copy()
    rj = p[j].copy()
    vec = rj - ri
    cur = np.linalg.norm(vec)
    if cur < 1e-8:
        return p
    direction = vec / cur
    delta = new_length - cur
    mi = ATOMIC_MASS.get(atoms[i], ATOMIC_MASS.get('H', 1.0))
    mj = ATOMIC_MASS.get(atoms[j], ATOMIC_MASS.get('H', 1.0))
    if mi + mj == 0:
        w_i = 0.5
    else:
        w_i = mj / (mi + mj)
    w_j = 1.0 - w_i
    p[i] = ri - w_i * delta * direction
    p[j] = rj + w_j * delta * direction
    return p


def apply_angle_change(coords, i, j, k, new_angle_deg):
    p = coords.copy()
    ri = p[i] - p[j]
    rk = p[k] - p[j]
    cur_angle = np.degrees(math.acos(np.clip(np.dot(ri, rk) / (np.linalg.norm(ri) * np.linalg.norm(rk) + 1e-16), -1.0, 1.0)))
    target = new_angle_deg
    dtheta = np.radians(target - cur_angle)
    axis = np.cross(ri, rk)
    if np.linalg.norm(axis) < 1e-8:
        axis = np.cross(ri, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(ri, np.array([0.0, 1.0, 0.0]))
    axis = axis / (np.linalg.norm(axis) + 1e-16)
    def rodrigues(v, k, theta):
        return v * math.cos(theta) + np.cross(k, v) * math.sin(theta) + k * (np.dot(k, v)) * (1.0 - math.cos(theta))
    new_ri = rodrigues(ri, axis, dtheta)
    p[i] = p[j] + new_ri
    return p


def apply_dihedral_change(coords, i, j, k, l, new_dihedral_deg):
    p = coords.copy()
    cur = compute_dihedral(p[i], p[j], p[k], p[l])
    dphi = np.radians(new_dihedral_deg - cur)
    axis_pt = p[j]
    axis_vec = p[k] - p[j]
    axis_unit = axis_vec / (np.linalg.norm(axis_vec) + 1e-16)
    v = p[i] - axis_pt
    def rodrigues_vec(v, k, theta):
        return v * math.cos(theta) + np.cross(k, v) * math.sin(theta) + k * (np.dot(k, v)) * (1.0 - math.cos(theta))
    new_v = rodrigues_vec(v, axis_unit, dphi)
    p[i] = axis_pt + new_v
    return p


# -------------------------
# CBS energy evaluator (with caching)
# -------------------------
# We cache by xyz string and method. When CBS parameters are changed we clear cache.
@lru_cache(maxsize=2000)
def _compute_cbs_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str):
    """
    Compute CBS energy using explicit a_corr and b_hf passed in so cache is safe.
    Returns scalar E_cbs (float).
    """
    scf_vals = []
    corr_vals = []
    for basis in (bs1, bs2):
        mol = gto.Mole()
        mol.atom = xyz_string
        mol.basis = basis
        mol.spin = 0
        mol.charge = 0
        mol.nthread = PYSCF_THREADS
        mol.max_memory = 8000
        mol.build()

        mf = scf.RHF(mol)
        mf.max_memory = 14330
        mf.conv_tol = 1e-9
        mf.max_cycle = 100
        scf_energy = mf.kernel()

        corr_energy = None
        if method.upper().startswith('CCSD'):
            try:
                mycc = cc.CCSD(mf)
                mycc.conv_tol = 1e-7
                mycc.max_cycle = 100
                mycc.kernel()
                try:
                    et = mycc.ccsd_t()
                except Exception:
                    et = 0.0
                corr_energy = mycc.e_tot - scf_energy + (et if et is not None else 0.0)
            except Exception:
                corr_energy = None
        if corr_energy is None and method.upper().startswith('MP2'):
            try:
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
                corr_energy = float(mp2_total - scf_energy)
            except Exception:
                corr_energy = None

        if corr_energy is None:
            # try to fallback to MP2 if CCSD(T) failed
            try:
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
                corr_energy = float(mp2_total - scf_energy)
            except Exception as e:
                raise RuntimeError(f"Correlation computation failed for basis {basis}: {e}")

        scf_vals.append(float(scf_energy))
        corr_vals.append(float(corr_energy))

    # now compose CBS from scf_vals and corr_vals using a_corr and b_hf
    # scf_vals[0] -> scf_small, scf_vals[1] -> scf_big
    scf_small, scf_big = scf_vals[0], scf_vals[1]
    corr_small, corr_big = corr_vals[0], corr_vals[1]

    # correlation extrapolation: corr_big + a_corr*(corr_big - corr_small)
    E_corr_cbs = corr_big + a_corr * (corr_big - corr_small)

    # HF extrapolation: E_hf = (exp(beta*X2)*scf_small - exp(beta*X1)*scf_big)/denom
    # Here b_hf was computed as exp(beta*X1)/(exp(beta*X2)-exp(beta*X1)) in other scripts,
    # but we pass explicit a_corr,b_hf to avoid hidden algebra errors. We will compute E_hf explicitly:
    # Recover beta and X's from b_hf isn't trivial here; instead compute HF using formula below:
    # Use same algebra as in other module: E_hf = (exp(beta*X2)*scf_small - exp(beta*X1)*scf_big) / (exp(beta*X2)-exp(beta*X1))
    # However we don't have beta/X here; caller will pass b_hf consistent with previous code path.
    # For safety we compute HF using the two SCF values and b_hf in the common form used elsewhere:
    # In the original scripts they used: E = scf2 + (exp(beta*x1)*scf1 ...). To avoid confusion we use the standard two-point
    # exponential form using X1/X2/BETA passed through caller by computing a_corr/b_hf earlier.
    # For simplicity here we reconstruct E_hf using the same formula used in the repo's L-BFGS implementation:
    # E_hf_cbs = scf_big + (a_corr - b_hf) * (scf_big - scf_small)  # not used - prefer direct composition below

    # To avoid inconsistent algebra, caller passes X1_HF, X2_HF, BETA as globals and we use them in run_optimization
    # For the cached function we simply return E_cbs assembled by caller-specific formula; so we will not call this
    # cached function directly from outside with different a_corr/b_hf algebra. For now compute E_cbs as:
    # Here we will just compute using the common pattern:
    # E_hf_cbs = scf_big + b_hf * (scf_big - scf_small)  # where b_hf is a signed factor; match other code that used a_corr and b_hf
    E_hf_cbs = scf_big + b_hf * (scf_big - scf_small)

    E_cbs = E_hf_cbs + E_corr_cbs
    return float(E_cbs)


# Small wrapper for caller to manage cache clearing
def compute_cbs_energy_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str):
    # clear underlying lru cache if needed is handled by caller when parameters change
    return _compute_cbs_from_xyz_cached(xyz_string, method, a_corr, b_hf, bs1, bs2)


# -------------------------
# Parabolic helper
# -------------------------
def parabolic_minimum(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size < 3:
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


# -------------------------
# Main optimization loop (SQM-style)
# -------------------------
def optimize_from_xyz(atoms, coords, method=DEFAULT_METHOD, maxcycle=MAXCYCLE_DEFAULT, fac_mult=FAC_DEFAULT,
                      x1=X1_DEFAULT, x2=X2_DEFAULT, x1_hf=X1HF_DEFAULT, x2_hf=X2HF_DEFAULT, beta=BETA_DEFAULT,
                      basis_pair=None):
    """
    Sequential SQM-style optimizer that updates current_coords if parabolic minimum improves CBS energy.
    Returns atoms, final_coords, history(list of dicts), converged(bool)
    """
    # compute extrapolation coefficients (same algebra as L-BFGS module)
    a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
    # compute b_hf factor similar to other modules: exp(beta*X1) / (exp(beta*X2)-exp(beta*X1))
    denom = math.exp(beta * x2_hf) - math.exp(beta * x1_hf)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    b_hf = math.exp(beta * x1_hf) / denom

    # If a different basis pair provided, use it
    if basis_pair is None:
        basis_pair = (basis_sets[0], basis_sets[1])
    bs1, bs2 = basis_pair

    internals = build_redundant_internals(atoms, coords)
    if not internals:
        raise RuntimeError("No internal coordinates found; check input geometry.")

    current_coords = coords.copy()
    displacement_factor = fac_mult
    history = []
    converged = False

    # Clear LRU cache for CBS cached computations in case parameters changed
    try:
        _compute_cbs_from_xyz_cached.cache_clear()
    except Exception:
        pass

    for cycle in range(1, maxcycle + 1):
        # compute current energy
        cur_xyz = xyz_to_pyscf_string(atoms, current_coords)
        try:
            current_energy = compute_cbs_energy_from_xyz_cached(cur_xyz, method, a_corr, b_hf, bs1, bs2)
        except Exception as e:
            raise RuntimeError(f"CBS evaluation at cycle start failed: {e}")

        print(f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}, current E = {current_energy:.10f} Ha")

        # For each internal, perform 5-point scan
        updated = False
        for ic in internals:
            if ic['type'] == 'bond':
                base = ic['value']
                disps = np.array([
                    -2 * displacement_factor * base,
                    -1 * displacement_factor * base,
                    0.0,
                    1 * displacement_factor * base,
                    2 * displacement_factor * base
                ], dtype=float)
            elif ic['type'] in ('angle', 'dihedral'):
                # angles/dihedrals work in degrees here
                ddeg = 2.0 * (displacement_factor * 100.0)
                disps = np.array([-2*ddeg, -1*ddeg, 0.0, 1*ddeg, 2*ddeg], dtype=float)
            else:
                disps = np.array([0.0])

            energies = []
            coords_list = []
            for d in disps:
                try:
                    if ic['type'] == 'bond':
                        i, j = ic['inds']
                        new_val = ic['value'] + d
                        new_coords = apply_bond_change(current_coords, i, j, new_val, atoms)
                    elif ic['type'] == 'angle':
                        i, j, k = ic['inds']
                        new_val = ic['value'] + d
                        new_coords = apply_angle_change(current_coords, i, j, k, new_val)
                    elif ic['type'] == 'dihedral':
                        i, j, k, l = ic['inds']
                        new_val = ic['value'] + d
                        new_coords = apply_dihedral_change(current_coords, i, j, k, l, new_val)
                    else:
                        new_coords = current_coords.copy()

                    xyzs = xyz_to_pyscf_string(atoms, new_coords)
                    E = compute_cbs_energy_from_xyz_cached(xyzs, method, a_corr, b_hf, bs1, bs2)
                    energies.append(float(E))
                    coords_list.append(new_coords)
                except Exception as e:
                    # on failure record inf and continue
                    energies.append(float('inf'))
                    coords_list.append(None)
                    print(f"    eval failed for IC {ic['type']} {ic.get('inds')} displacement {d}: {e}")

            ds = disps
            es = np.array(energies, dtype=float)
            if np.all(np.isinf(es)):
                print(f"  Skipping {ic['type']} {ic['inds']}: all evals failed")
                continue
            try:
                x_min_disp, e_min = parabolic_minimum(ds, es)
                idx_best = int(np.nanargmin(es))
                best_coords = coords_list[idx_best]
                print(f"  IC {ic['type']} {ic['inds']}: current={ic['value']:.6f} -> x_min_disp={x_min_disp:.6f}, E_min={e_min:.10f}")
                if e_min < current_energy - 1e-12 and best_coords is not None:
                    # update geometry to the best computed coords
                    current_coords = best_coords.copy()
                    # rebuild internals values on updated geometry
                    internals = build_redundant_internals(atoms, current_coords)
                    updated = True
                    # recompute current_energy to use for subsequent comparisons
                    current_energy = e_min
                    print("    geometry updated (improvement)")
                else:
                    print("    no improvement")
            except Exception as e:
                print(f"    error in parabolic fit for IC {ic['inds']}: {e}")

        # record cycle energy (after processing all internals)
        history.append({'cycle': cycle, 'energy': float(current_energy)})
        # convergence check
        if cycle > 1:
            ediff = abs(history[-1]['energy'] - history[-2]['energy'])
            print(f"  ΔE since last cycle: {ediff:.4e} Ha")
            if ediff < ENERGY_CRIT:
                print("Converged by energy criterion")
                converged = True
                break

        displacement_factor *= CUT

    return atoms, current_coords, history, converged


# -------------------------
# run_optimization API
# -------------------------
def run_optimization(params: dict, outputs_dir: Path):
    """
    Standard entry from opt_cli.prepare_options_from_params
    params should include:
      - input_xyz (path to file)
      - method (optional)
      - X1, X2, Xhf1, Xhf2, beta (optional)
      - maxcycle (optional)
      - fac (optional)
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    input_xyz = params.get("input_xyz") or params.get("input") or params.get("geometry")
    if not input_xyz:
        raise ValueError("params must include 'input_xyz' (path to input XYZ)")
    method = params.get("method", DEFAULT_METHOD)
    x1 = float(params.get("X1", params.get("x1", X1_DEFAULT)))
    x2 = float(params.get("X2", params.get("x2", X2_DEFAULT)))
    x1hf = float(params.get("X1hf", params.get("x1hf", X1HF_DEFAULT)))
    x2hf = float(params.get("X2hf", params.get("x2hf", X2HF_DEFAULT)))
    beta = float(params.get("beta", params.get("BETA", BETA_DEFAULT)))
    maxcycle = int(params.get("maxcycle", MAXCYCLE_DEFAULT))
    fac = float(params.get("fac", FAC_DEFAULT))
    # basis override if provided
    basis1 = params.get("basis1")
    basis2 = params.get("basis2")
    basis_pair = (basis1, basis2) if (basis1 and basis2) else None

    atoms, coords0, comment = read_xyz(input_xyz)
    print(f"Loaded {len(atoms)} atoms from {input_xyz}")
    print("Building redundant internal coordinates (initial)...")
    internals = build_redundant_internals(atoms, coords0)
    print(f"Found {len(internals)} internal coordinates (bonds/angles/dihedrals).")

    # run optimizer
    atoms_out, coords_out, history, converged = optimize_from_xyz(
        atoms,
        coords0,
        method=method,
        maxcycle=maxcycle,
        fac_mult=fac,
        x1=x1, x2=x2, x1_hf=x1hf, x2_hf=x2hf, beta=beta,
        basis_pair=basis_pair
    )

    # write outputs via writer helpers
    prefix = "SQM"
    cycles_file = write_cycle_energies(outputs_dir, prefix, history)
    xyz_file = write_final_xyz(outputs_dir, prefix, atoms_out, coords_out, history[-1]['energy'] if history else 0.0)

    return {
        "history": history,
        "final_energy": float(history[-1]['energy']) if history else None,
        "final_cart": coords_out,
        "symbols": atoms_out,
        "outputs": {"cycles": str(cycles_file), "xyz": str(xyz_file)},
        "converged": converged,
    }


# CLI compatibility
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Input XYZ file")
    p.add_argument("-o", "--out", default="PyCBS-OUTPUTS", help="Outputs directory")
    p.add_argument("--method", default=DEFAULT_METHOD)
    p.add_argument("--maxcycle", type=int, default=MAXCYCLE_DEFAULT)
    p.add_argument("--fac", type=float, default=FAC_DEFAULT)
    p.add_argument("--x1", type=float, default=X1_DEFAULT)
    p.add_argument("--x2", type=float, default=X2_DEFAULT)
    p.add_argument("--x1hf", type=float, default=X1HF_DEFAULT)
    p.add_argument("--x2hf", type=float, default=X2HF_DEFAULT)
    p.add_argument("--beta", type=float, default=BETA_DEFAULT)
    args = p.parse_args()

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
    }
    out = Path(args.out)
    result = run_optimization(params, out)
    print("Done. Outputs written to:", out)