#!/usr/bin/env python3
"""
pyCBS_ric.py

Flexible pyCBS optimizer: read an XYZ, build redundant internal coordinates
(bonds, angles, dihedrals), perform per-internal-coordinate 5-point parabolic
optimization using CBS-extrapolated energies (MP2 or CCSD(T)), and write an
optimized XYZ.

Usage:
    python pyCBS_ric.py -i input.xyz [--method CCSD(T)|MP2] [--maxcycle 20] [--workers 4]
    You can also pass CBS/extrapolation parameters:
      --x1 1.85 --x2 2.639 --x1hf 3.02 --x2hf 3.64 --beta 1.62

Notes:
- Internal->Cartesian updates are geometric and approximate (no Wilson B-matrix).
- For robust/large-molecule optimization prefer a full internal-coordinate optimizer.
"""
import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from functools import lru_cache
from tqdm import tqdm

import numpy as np
from pyscf import gto, scf, cc, lib, mp
from multiprocessing import cpu_count

# -------------------------
# Resources / parallelism
# -------------------------
DEFAULT_MAX_WORKERS = max(1, cpu_count() - 1)
PYSCF_THREADS = max(1, DEFAULT_MAX_WORKERS // 2)

os.environ['MKL_NUM_THREADS'] = str(PYSCF_THREADS)
os.environ['OMP_NUM_THREADS'] = str(PYSCF_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(PYSCF_THREADS)
lib.num_threads(PYSCF_THREADS)

# -------------------------
# Covalent radii and approximate masses (a few elements; extend as needed)
# -------------------------
COV_RAD = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02
}
ATOMIC_MASS = {
    'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.999, 'F': 18.998,
    'P': 30.9738, 'S': 32.065, 'Cl': 35.453
}

# -------------------------
# CBS / extrapolation & defaults (kept from your script)
# -------------------------
basis_sets = ['cc-pvtz', 'cc-pvqz']
METHOD = 'CCSD(T)'
# default CBS/extrapolation parameters (can be overridden via CLI)
x1_default, x2_default = 2.792, 3.719
x1hf_default, x2hf_default = 3.64, 4.28
beta_default = 1.62

# these will be set from CLI into globals x1, x2, x1_hf, x2_hf, beta and then used
x1 = x1_default
x2 = x2_default
x1_hf = x1hf_default
x2_hf = x2hf_default
beta = beta_default

# compute extrapolation constants (will be recomputed in main after parsing args)
a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
b_hf = (np.exp(beta * x1_hf)) / (np.exp(beta * x2_hf) - np.exp(beta * x1_hf))

maxcycle_default = 20
energy_criterion = 1e-8
fac_mult_default = 0.05
cut = 0.75

# -------------------------
# Utility: file I/O, geometry helpers
# -------------------------
def read_xyz(filename):
    with open(filename) as f:
        natoms = int(f.readline().strip())
        comment = f.readline().rstrip('\n')
        atoms = []
        coords = []
        for _ in range(natoms):
            parts = f.readline().split()
            if len(parts) < 4:
                continue
            symbol = parts[0]
            x, y, z = map(float, parts[1:4])
            atoms.append(symbol)
            coords.append([x, y, z])
    return atoms, np.array(coords, dtype=float), comment

def write_xyz(filename, atoms, coords, comment="Optimized geometry"):
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for a, c in zip(atoms, coords):
            f.write(f"{a} {c[0]: .8f} {c[1]: .8f} {c[2]: .8f}\n")

def xyz_to_pyscf_string(atoms, coords):
    lines = []
    for a, c in zip(atoms, coords):
        lines.append(f"{a} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}")
    return "\n".join(lines)

def distance(a, b):
    return np.linalg.norm(a - b)

# -------------------------
# Internal coordinate builder (redundant set)
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
        for i_idx in range(len(neighbors)):
            for k_idx in range(i_idx + 1, len(neighbors)):
                i = neighbors[i_idx]
                k = neighbors[k_idx]
                angles.append((i, j, k))
    dihedrals = []
    # Build dihedrals i-j-k-l where (i,j),(j,k),(k,l) are bonds
    bond_set = set(bonds)
    for (j, k) in bonds:
        neigh_j = [i for i, b in bonds if b == j and i != k] + [b for a, b in bonds if a == j and b != k]
        neigh_k = [i for i, b in bonds if b == k and i != j] + [b for a, b in bonds if a == k and b != j]
        neigh_j = list(set(neigh_j))
        neigh_k = list(set(neigh_k))
        for i in neigh_j:
            for l in neigh_k:
                if len({i, j, k, l}) == 4:
                    dihedrals.append((i, j, k, l))
    # Remove duplicates (normalize tuples)
    bonds = sorted(set(bonds))
    angles = sorted(set(angles))
    dihedrals = sorted(set(dihedrals))
    internals = []
    # encode internal coords as dicts: type, indices, value (Å or deg)
    for (i, j) in bonds:
        internals.append({'type': 'bond', 'inds': (i, j)})
    for (i, j, k) in angles:
        internals.append({'type': 'angle', 'inds': (i, j, k)})
    for (i, j, k, l) in dihedrals:
        internals.append({'type': 'dihedral', 'inds': (i, j, k, l)})
    # compute current values
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
# Internal -> Cartesian transformations (approximate, geometric)
# -------------------------
def apply_bond_change(coords, i, j, new_length):
    """Move i and j along the bond vector to set bond length to new_length.
       Distribute displacements inversely proportional to approximate masses."""
    p = coords.copy()
    ri = p[i].copy()
    rj = p[j].copy()
    vec = rj - ri
    cur = np.linalg.norm(vec)
    if cur < 1e-8:
        return p
    direction = vec / cur
    delta = new_length - cur
    mi = ATOMIC_MASS.get('H', 1.0)
    mj = ATOMIC_MASS.get('H', 1.0)
    mi = ATOMIC_MASS.get(args_atoms[i], mi) if 'args_atoms' in globals() else ATOMIC_MASS.get('H', 1.0)
    mj = ATOMIC_MASS.get(args_atoms[j], mj) if 'args_atoms' in globals() else ATOMIC_MASS.get('H', 1.0)
    if mi + mj == 0:
        w_i = 0.5
    else:
        w_i = mj / (mi + mj)
    w_j = 1.0 - w_i
    p[i] = ri - w_i * delta * direction
    p[j] = rj + w_j * delta * direction
    return p

def apply_angle_change(coords, i, j, k, new_angle_deg):
    """Rotate atom i or k around central j to set the angle i-j-k to new_angle.
       Here, we rotate atom i (keeping j and k fixed) for simplicity."""
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
    """Rotate the fragment i...j around bond j-k to set dihedral i-j-k-l.
       For simplicity rotate atom i around axis j->k (keeping other atoms fixed)."""
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

# -------------------------
# CBS energy evaluator from XYZ (cached)
# -------------------------
def raw_energy(ex1, ex2, ex1hf, ex2hf):
    return ex2 + a_corr * (ex2 - ex1) + (a_corr - b_hf) * (ex2hf - ex1hf)

@lru_cache(maxsize=2000)
def compute_cbs_energy_from_xyz_cached(xyz_string, method):
    """xyz_string: string with 'Element X Y Z' lines separated by newlines."""
    results = {'scf': [], 'corr': []}
    for basis in basis_sets:
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

        if method == 'CCSD(T)':
            mycc = cc.CCSD(mf)
            mycc.conv_tol = 1e-7
            mycc.max_cycle = 100
            mycc.kernel()
            try:
                et = mycc.ccsd_t()
            except Exception:
                et = None
            corr_energy = mycc.e_tot + (et if et is not None else 0.0)
        elif method == 'MP2':
            try:
                mymp = mp.MP2(mf)
                mymp.max_memory = 14330
                mp_res = mymp.run()
                mp2_total = getattr(mp_res, 'e_tot', None)
                if mp2_total is None:
                    mp2_total = getattr(mymp, 'e_tot', None)
                if mp2_total is None:
                    e_corr = getattr(mp_res, 'e_corr', getattr(mymp, 'e_corr', None))
                    if e_corr is not None:
                        mp2_total = scf_energy + e_corr
                if mp2_total is None:
                    raise RuntimeError("Could not retrieve MP2 total energy")
                corr_energy = float(mp2_total)
            except Exception as e:
                raise RuntimeError(f"MP2 failed: {e}")
        else:
            raise ValueError(f"Unknown method '{method}'")
        results['scf'].append(float(scf_energy))
        results['corr'].append(float(corr_energy))
    return raw_energy(results['corr'][0], results['corr'][1], results['scf'][0], results['scf'][1])

def compute_cbs_energy_from_xyz(atoms, coords, method):
    xyz_str = xyz_to_pyscf_string(atoms, coords)
    return compute_cbs_energy_from_xyz_cached(xyz_str, method)

# -------------------------
# Per-internal displacement evaluation (for a single coordinate)
# -------------------------
def evaluate_internal_displacements(args):
    """Worker: given atoms, coords, an internal coordinate and a list of displacements,
       return the displacement values and energies arrays."""
    atoms, coords, ic, displacements, method = args
    results = []
    for d in displacements:
        try:
            if ic['type'] == 'bond':
                i, j = ic['inds']
                new_val = ic['value'] + d
                new_coords = apply_bond_change(coords, i, j, new_val)
            elif ic['type'] == 'angle':
                i, j, k = ic['inds']
                new_val = ic['value'] + d
                new_coords = apply_angle_change(coords, i, j, k, new_val)
            elif ic['type'] == 'dihedral':
                i, j, k, l = ic['inds']
                new_val = ic['value'] + d
                new_coords = apply_dihedral_change(coords, i, j, k, l, new_val)
            else:
                continue
            energy = compute_cbs_energy_from_xyz_cached(xyz_to_pyscf_string(atoms, new_coords), method)
            results.append((d, energy, new_coords))
        except Exception:
            results.append((d, float('inf'), None))
    ds = np.array([r[0] for r in results])
    es = np.array([r[1] for r in results])
    return ic, ds, es, results

# -------------------------
# Parabolic fit helper
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
# Optimization driver (redundant internal coords)
# -------------------------
def optimize_from_xyz(atoms, coords, method=METHOD, maxcycle=maxcycle_default, workers=DEFAULT_MAX_WORKERS, fac_mult=fac_mult_default):
    global args_atoms
    args_atoms = atoms  # used inside apply_xxx functions to look up masses
    internals = build_redundant_internals(atoms, coords)
    if not internals:
        raise RuntimeError("No internal coordinates found; check input geometry.")
    current_coords = coords.copy()
    displacement_factor = fac_mult
    history = []
    converged = False

    executor = ProcessPoolExecutor(max_workers=workers)

    try:
        for cycle in range(1, maxcycle + 1):
            print(f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}")
            tasks = []
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
                    ddeg = 2.0 * (displacement_factor * 100.0)
                    disps = np.array([-2*ddeg, -1*ddeg, 0.0, 1*ddeg, 2*ddeg], dtype=float)
                else:
                    disps = np.array([0.0])
                tasks.append((atoms, current_coords.copy(), ic.copy(), disps, method))

            futures = [executor.submit(evaluate_internal_displacements, t) for t in tasks]
            results = []
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print("Worker error:", e)
            for ic, ds, es, detailed in results:
                if np.all(np.isinf(es)):
                    print(f"  Skipping {ic['type']} {ic['inds']}: all evals failed")
                    continue
                try:
                    x_min, e_min = parabolic_minimum(ds, es)
                    idx_best = int(np.argmin(es))
                    best_coords = detailed[idx_best][2] if detailed[idx_best][2] is not None else None
                    print(f"  IC {ic['type']} {ic['inds']}: current={ic['value']:.6f} -> x_min_disp={x_min:.6f}, E_min={e_min:.10f}")
                    current_energy = compute_cbs_energy_from_xyz_cached(xyz_to_pyscf_string(atoms, current_coords), method)
                    if e_min < current_energy - 1e-12 and best_coords is not None:
                        current_coords = best_coords.copy()
                        internals = build_redundant_internals(atoms, current_coords)
                        print("    geometry updated (improvement)")
                    else:
                        print("    no improvement")
                except Exception as e:
                    print("    error in parabolic fit:", e)

            cur_e = compute_cbs_energy_from_xyz_cached(xyz_to_pyscf_string(atoms, current_coords), method)
            history.append({'cycle': cycle, 'coords': current_coords.copy(), 'energy': cur_e})
            if cycle > 1:
                ediff = abs(history[-1]['energy'] - history[-2]['energy'])
                print(f"  ΔE since last cycle: {ediff:.4e} Ha")
                if ediff < energy_criterion:
                    print("Converged by energy criterion")
                    converged = True
                    break
            displacement_factor *= cut
    finally:
        executor.shutdown(wait=True)

    return atoms, current_coords, history, converged

# -------------------------
# CLI
# -------------------------
def main():
    global x1, x2, x1_hf, x2_hf, beta, a_corr, b_hf
    parser = argparse.ArgumentParser(description="pyCBS redundant-internal-coordinate optimizer")
    parser.add_argument('-i', '--input', required=True, help="Input XYZ file")
    parser.add_argument('-o', '--output', default='optimized.xyz', help="Output XYZ file")
    parser.add_argument('--method', default=METHOD, choices=['CCSD(T)', 'MP2'], help="Correlation method")
    parser.add_argument('--basis_set', default=basis_sets, choices=['cc-pvdz','cc-pvtz', 'cc-pvqz'], help="Basis set to perform CBS extrapolations")
    parser.add_argument('--maxcycle', type=int, default=maxcycle_default, help="Max optimization cycles")
    parser.add_argument('--workers', type=int, default=DEFAULT_MAX_WORKERS, help="Number of parallel workers")
    parser.add_argument('--fac', type=float, default=fac_mult_default, help="Initial fractional displacement factor")
    parser.add_argument('--x1', type=float, default=x1_default, help=f"x1 for correlation extrapolation (default {x1_default})")
    parser.add_argument('--x2', type=float, default=x2_default, help=f"x2 for correlation extrapolation (default {x2_default})")
    parser.add_argument('--x1hf', type=float, default=x1hf_default, help=f"x1_hf for HF extrapolation (default {x1hf_default})")
    parser.add_argument('--x2hf', type=float, default=x2hf_default, help=f"x2_hf for HF extrapolation (default {x2hf_default})")
    parser.add_argument('--beta', type=float, default=beta_default, help=f"beta for HF extrapolation (default {beta_default})")
    args = parser.parse_args()

    # assign globals for extrapolation and recompute coefficients
    x1 = args.x1
    x2 = args.x2
    x1_hf = args.x1hf
    x2_hf = args.x2hf
    beta = args.beta
    a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
    b_hf = (np.exp(beta * x1_hf)) / (np.exp(beta * x2_hf) - np.exp(beta * x1_hf))

    atoms, coords, comment = read_xyz(args.input)
    print(f"Loaded {len(atoms)} atoms from {args.input}")
    print("Building redundant internal coordinates...")
    internals = build_redundant_internals(atoms, coords)
    print(f"Found {len(internals)} internal coordinates (bonds/angles/dihedrals).")

    atoms_out, coords_out, history, conv = optimize_from_xyz(atoms, coords, method=args.method, maxcycle=args.maxcycle, workers=args.workers, fac_mult=args.fac)

    write_xyz(args.output, atoms_out, coords_out, comment=f"Optimized by pyCBS ({args.method})")
    print(f"Optimized geometry written to {args.output}")
    final_e = compute_cbs_energy_from_xyz_cached(xyz_to_pyscf_string(atoms_out, coords_out), args.method)
    print(f"Final CBS energy: {final_e:.10f} Ha")

if __name__ == "__main__":
    main()