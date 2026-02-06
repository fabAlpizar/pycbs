#!/usr/bin/env python3
"""
SQM-based optimizer module — 3-point sampling + ProcessPoolExecutor parallelism.

Usage: same as your previous SQM module. New params:
  - workers (int): number of processes for parallel sampling/validation (default: cpu_count() or 1)
  - top_k (int): how many predicted minima to validate per subiteration (default: 1)

Important notes about PySCF + multiprocessing:
 - Each worker process will set OMP/MKL/OPENBLAS env vars and call pyscf.lib.num_threads(1).
 - Sharing caches between processes is not possible; each process keeps its own LRU cache.
 - This implementation parallelizes the expensive single-point CBS evaluations.
"""
from pathlib import Path
import math
import os
import sys
from functools import lru_cache
from multiprocessing import cpu_count
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pycbs.writer import write_cycle_energies, write_final_xyz

# PySCF imports (deferred errors will be raised if PySCF missing)
try:
    from pyscf import gto, scf, cc, mp, lib
except Exception as e:
    raise ImportError("PySCF is required for this script. Install pyscf and retry.") from e

# -------------------------
# Defaults / parameters
# -------------------------
basis_sets = ['cc-pvdz', 'cc-pvtz']
DEFAULT_METHOD = 'CCSD(T)'

X1_DEFAULT, X2_DEFAULT = 1.85, 2.639
X1HF_DEFAULT, X2HF_DEFAULT = 3.02, 3.64
BETA_DEFAULT = 1.62

# threads (main process default; worker processes will set their own)
PYSCF_THREADS = 1
lib.num_threads(PYSCF_THREADS)

MAXCYCLE_DEFAULT = 50
ENERGY_CRIT = 1e-8
FAC_DEFAULT = 0.05
CUT = 0.75

ATOMIC_MASS = {
    'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.999, 'F': 18.998,
    'P': 30.9738, 'S': 32.065, 'Cl': 35.453
}

COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02
}
DEFAULT_COV_RAD = 0.77

DEFAULT_SPIN = 0

# runtime defaults for parallelism
DEFAULT_WORKERS = max(1, cpu_count() - 1)
DEFAULT_TOP_K = 1

# thresholds for selection
ENERGY_ACCEPT_TOL = 1e-6  # per-move acceptance (Hartree)
MIN_CURVATURE = 1e-10     # minimal convex curvature required (Ha / unit^2)


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


def _angle_deg(a, b, c):
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-16)
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _dihedral_deg(p0, p1, p2, p3):
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= (np.linalg.norm(b1) + 1e-16)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _value_for_internal(kind: str, inds: tuple[int, ...], coords: np.ndarray) -> float:
    if kind == 'bond':
        i, j = inds
        return float(np.linalg.norm(coords[i] - coords[j]))
    elif kind == 'angle':
        i, j, k = inds
        return _angle_deg(coords[i], coords[j], coords[k])
    elif kind == 'dihedral':
        i, j, k, l = inds
        return _dihedral_deg(coords[i], coords[j], coords[k], coords[l])
    else:
        raise ValueError("Unknown internal type: " + str(kind))


def _label_internal(kind: str, inds: tuple[int, ...], atoms: list[str]) -> str:
    if kind == 'bond':
        i, j = inds
        return f"bond:{i+1}-{j+1}:{atoms[i]}-{atoms[j]}"
    if kind == 'angle':
        i, j, k = inds
        return f"angle:{i+1}-{j+1}-{k+1}:{atoms[i]}-{atoms[j]}-{atoms[k]}"
    if kind == 'dihedral':
        i, j, k, l = inds
        return f"dihedral:{i+1}-{j+1}-{k+1}-{l+1}:{atoms[i]}-{atoms[j]}-{atoms[k]}-{atoms[l]}"
    return f"intern:{inds}"


# -------------------------
# Geometry-derived internals (robust RIC manager)
# -------------------------
def _covalent_radius(elem: str) -> float:
    return COVALENT_RADII.get(elem, DEFAULT_COV_RAD)


def generate_internals_from_geometry(atoms: list[str], coords: np.ndarray, scale: float = 1.2):
    nat = len(atoms)
    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    bonds = set()
    for i in range(nat):
        for j in range(i + 1, nat):
            ri = _covalent_radius(atoms[i])
            rj = _covalent_radius(atoms[j])
            cutoff = scale * (ri + rj)
            if 0.2 < dmat[i, j] <= cutoff:
                bonds.add((i, j))
    bonds = sorted(bonds)
    bond_neighbors = {i: [] for i in range(nat)}
    for i, j in bonds:
        bond_neighbors[i].append(j)
        bond_neighbors[j].append(i)

    out = []
    for i, j in bonds:
        out.append({"type": "bond", "inds": (i, j), "value": float(dmat[i, j])})

    angles = set()
    for j in range(nat):
        neigh = bond_neighbors.get(j, [])
        if len(neigh) < 2:
            continue
        for ii in range(len(neigh)):
            for kk in range(ii + 1, len(neigh)):
                i = neigh[ii]
                k = neigh[kk]
                inds = (i, j, k)
                if inds not in angles:
                    angles.add(inds)
                    val = _angle_deg(coords[i], coords[j], coords[k])
                    out.append({"type": "angle", "inds": inds, "value": float(val)})

    diheds = set()
    for j in range(nat):
        for k in bond_neighbors.get(j, []):
            for i in bond_neighbors.get(j, []):
                if i == k:
                    continue
                for l in bond_neighbors.get(k, []):
                    if l == j:
                        continue
                    inds = (i, j, k, l)
                    if inds not in diheds:
                        diheds.add(inds)
                        val = _dihedral_deg(coords[i], coords[j], coords[k], coords[l])
                        out.append({"type": "dihedral", "inds": inds, "value": float(val)})

    if not out:
        raise RuntimeError("Fallback geometry-based internals generation produced no internals (geometry suspicious).")
    return out


# -------------------------
# Geometric updates (approximate local moves)
# -------------------------
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
    mi = ATOMIC_MASS.get(atoms[i], 1.0)
    mj = ATOMIC_MASS.get(atoms[j], 1.0)
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
    cur_angle = _angle_deg(p[i], p[j], p[k])
    dtheta = math.radians(new_angle_deg - cur_angle)
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
    cur = _dihedral_deg(p[i], p[j], p[k], p[l])
    dphi = math.radians(new_dihedral_deg - cur)
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
@lru_cache(maxsize=2000)
def _compute_cbs_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str):
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

    scf_small, scf_big = scf_vals[0], scf_vals[1]
    corr_small, corr_big = corr_vals[0], corr_vals[1]

    E_corr_cbs = corr_big + a_corr * (corr_big - corr_small)
    E_hf_cbs = scf_big + b_hf * (scf_big - scf_small)
    E_cbs = E_hf_cbs + E_corr_cbs
    return float(E_cbs)


def compute_cbs_energy_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str):
    return _compute_cbs_from_xyz_cached(xyz_string, method, a_corr, b_hf, bs1, bs2)


# -------------------------
# 3-point parabolic helper
# -------------------------
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


# -------------------------
# Worker function (picklable) for parallel evaluation of a single displacement
# -------------------------
def _worker_eval_disp(task):
    """
    task: tuple containing:
      (atoms:list, coords_list (nested list), ic_type:str, inds:tuple, base_value:float, disp:float,
       method:str, a_corr:float, b_hf:float, bs1:str, bs2:str, pyscf_threads:int)
    returns:
      (ic_index_dummy, disp, energy (float), coords_list (2D list))
    Note: we include ic_index_dummy as the first field so main can group results easily.
    """
    # Unpack
    (ic_index, atoms, coords_list, ic_type, inds, base_value, disp,
     method, a_corr, b_hf, bs1, bs2, pyscf_threads) = task

    # Ensure worker uses single-threaded linear algebra
    try:
        os.environ['OMP_NUM_THREADS'] = str(pyscf_threads)
        os.environ['MKL_NUM_THREADS'] = str(pyscf_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(pyscf_threads)
    except Exception:
        pass
    try:
        lib.num_threads(max(1, int(pyscf_threads)))
    except Exception:
        # if worker import of lib not available, continue (it should be)
        pass

    coords = np.asarray(coords_list, dtype=float)
    # build the new coordinates for this displacement
    try:
        if ic_type == 'bond':
            i, j = inds
            new_val = base_value + disp
            new_coords = apply_bond_change(coords, i, j, new_val, atoms)
        elif ic_type == 'angle':
            i, j, k = inds
            new_val = base_value + disp
            new_coords = apply_angle_change(coords, i, j, k, new_val)
        elif ic_type == 'dihedral':
            i, j, k, l = inds
            new_val = base_value + disp
            new_coords = apply_dihedral_change(coords, i, j, k, l, new_val)
        else:
            new_coords = coords.copy()
    except Exception as e:
        return (ic_index, disp, float('inf'), None, f"geom_apply_error: {e}")

    xyzs = xyz_to_pyscf_string(atoms, new_coords)
    try:
        E = compute_cbs_energy_from_xyz_cached(xyzs, method, a_corr, b_hf, bs1, bs2)
        return (ic_index, disp, float(E), new_coords.tolist(), None)
    except Exception as e:
        return (ic_index, disp, float('inf'), None, f"eval_error: {e}")


# -------------------------
# Main optimization loop (parallelized sampling + validations)
# -------------------------
def optimize_from_xyz(atoms, coords, method=DEFAULT_METHOD, maxcycle=MAXCYCLE_DEFAULT, fac_mult=FAC_DEFAULT,
                      x1=X1_DEFAULT, x2=X2_DEFAULT, x1_hf=X1HF_DEFAULT, x2_hf=X2HF_DEFAULT, beta=BETA_DEFAULT,
                      basis_pair=None, workers: int = DEFAULT_WORKERS, top_k: int = DEFAULT_TOP_K,
                      pyscf_threads_per_worker: int = 1, debug: bool = False):
    """
    Parallel SQM-style optimizer (3-point sampling). The `workers` and `top_k` parameters
    control parallelism and how many predicted minima are validated per subiteration.
    """
    a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
    denom = math.exp(beta * x2_hf) - math.exp(beta * x1_hf)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    b_hf = math.exp(beta * x1_hf) / denom

    if basis_pair is None:
        basis_pair = (basis_sets[0], basis_sets[1])
    bs1, bs2 = basis_pair

    baseline_internals = generate_internals_from_geometry(atoms, coords)
    if not baseline_internals:
        raise RuntimeError("No internal coordinates found; check input geometry.")

    baseline_labels = [_label_internal(ic['type'], ic['inds'], atoms) for ic in baseline_internals]
    baseline_inds = [ic['inds'] for ic in baseline_internals]
    baseline_types = [ic['type'] for ic in baseline_internals]

    internals = [dict(ic) for ic in baseline_internals]

    current_coords = coords.copy()
    displacement_factor = fac_mult
    history = []
    converged = False
    internals_trace = []
    init_vals = {lbl: _value_for_internal(tp, inds, current_coords) for lbl, tp, inds in zip(baseline_labels, baseline_types, baseline_inds)}
    internals_trace.append(init_vals)

    # clear cache in main process
    try:
        _compute_cbs_from_xyz_cached.cache_clear()
    except Exception:
        pass

    # Create executor (shared across cycles); workers will be reused
    workers = max(1, int(workers))
    if workers == 1:
        # we'll still use ProcessPoolExecutor with max_workers=1 to keep code paths consistent
        pass

    for cycle in range(1, maxcycle + 1):
        cur_xyz = xyz_to_pyscf_string(atoms, current_coords)
        try:
            current_energy = compute_cbs_energy_from_xyz_cached(cur_xyz, method, a_corr, b_hf, bs1, bs2)
        except Exception as e:
            raise RuntimeError(f"CBS evaluation at cycle start failed: {e}")

        print(f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}, current E = {current_energy:.10f} Ha")

        applied_changes = []

        # Best-improvement subiterations: repeat until no acceptable candidate remains
        while True:
            # PHASE 1: create tasks (one per internal per displacement) and submit to executor
            tasks = []
            ic_ds_map = {}  # store ds ordering per internal index
            for idx, ic in enumerate(internals):
                tp = ic['type']; inds = ic['inds']
                base_value = ic.get('value', _value_for_internal(tp, inds, current_coords))
                if tp == 'bond':
                    h = displacement_factor * base_value
                    ds = np.array([-h, 0.0, h], dtype=float)
                elif tp in ('angle', 'dihedral'):
                    h = 2.0 * (displacement_factor * 100.0)
                    ds = np.array([-h, 0.0, h], dtype=float)
                else:
                    ds = np.array([0.0])
                ic_ds_map[idx] = (ds, tp, inds, base_value)
                for d in ds:
                    tasks.append((idx, atoms, current_coords.tolist(), tp, inds, base_value, float(d),
                                  method, a_corr, b_hf, bs1, bs2, int(pyscf_threads_per_worker)))

            # collect results
            sampled = {i: [] for i in range(len(internals))}  # idx -> list of (d, E, coords, err)
            if tasks:
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
                    futures = {ex.submit(_worker_eval_disp, t): t for t in tasks}
                    for fut in as_completed(futures):
                        res = fut.result()
                        # res is (ic_index, disp, energy, coords_list or None, err_msg or None)
                        ic_index, disp, energy, coords_out, err = res
                        sampled[ic_index].append((disp, energy, coords_out, err))

            # Now process each internal's sampled values and compute predicted minima
            candidates = []  # list of candidate dicts to validate {idx, predicted_disp, e_pred, curvature, sampled_best,...}
            for idx, (ds, tp, inds, base_value) in ic_ds_map.items():
                samples = sampled.get(idx, [])
                if len(samples) == 0:
                    # no samples (all failed)
                    if debug:
                        print(f"  IC {tp} {inds}: no successful samples (all evals failed)")
                    continue
                # reorder samples to match ds ordering
                # ds is [-h,0,h] order
                # create mapping from disp->(E, coords, err)
                map_by_d = {float(s[0]): s for s in samples}
                try:
                    es = np.array([float(map_by_d[float(d)][1]) for d in ds], dtype=float)
                    coords_list = [map_by_d[float(d)][2] for d in ds]
                    errs = [map_by_d[float(d)][3] for d in ds]
                except Exception:
                    # If some displacements missing, fallback to available ones
                    available = sorted(samples, key=lambda x: x[0])
                    es = np.array([s[1] for s in available], dtype=float)
                    coords_list = [s[2] for s in available]
                    errs = [s[3] for s in available]
                    # ensure we have at least one valid point
                    if es.size == 0 or np.all(np.isinf(es)):
                        if debug:
                            print(f"  IC {tp} {inds}: all sampled energies invalid")
                        continue
                    # If not 3 points we still can use np.argmin as fallback
                    if es.size < 3:
                        idx_best = int(np.nanargmin(es))
                        candidates.append({
                            "idx": idx,
                            "which": "sampled_only",
                            "sampled_best_disp": available[idx_best][0],
                            "sampled_best_energy": float(available[idx_best][1]),
                            "sampled_best_coords": available[idx_best][2],
                            "curvature": None,
                            "pred_disp": None,
                            "pred_energy": None
                        })
                        continue

                if np.all(np.isinf(es)):
                    if debug:
                        print(f"  IC {tp} {inds}: all sample energies inf (SCF fails)")
                    continue

                # compute parabolic 3-pt fit
                x_min_disp, e_min = parabolic_minimum_3pt(ds, es)
                # numeric curvature and grad
                Eminus, E0, Eplus = es[0], es[1], es[2]
                hstep = float(ds[2] - ds[1])
                grad0 = (Eplus - Eminus) / (2.0 * hstep)
                curvature = (Eplus + Eminus - 2.0 * E0) / (hstep * hstep)

                # sampled best
                idx_best = int(np.nanargmin(es))
                sampled_best_energy = float(es[idx_best])
                sampled_best_coords = coords_list[idx_best]

                # predicted improvement relative to current_energy
                deltaE_pred = current_energy - float(e_min)

                if debug:
                    cur_val = base_value
                    print(f"  IC {tp} {inds}: cur={cur_val:.6f}, ds=[{ds[0]:.6g}, 0, {ds[2]:.6g}], E_samples=[{es[0]:.6f}, {es[1]:.6f}, {es[2]:.6f}], x_min_disp={x_min_disp:.6f}, e_min={e_min:.10f}, curvature={curvature:.3e}")

                # selection: curvature positive, predicted improvement > tol
                if curvature is not None and curvature > MIN_CURVATURE and deltaE_pred > ENERGY_ACCEPT_TOL:
                    # candidate for validation: predicted geometry (may lie between sample points)
                    candidates.append({
                        "idx": idx,
                        "which": "predicted",
                        "pred_disp": x_min_disp,
                        "pred_energy_est": float(e_min),
                        "curvature": float(curvature),
                        "grad0": float(grad0),
                        "sampled_best_disp": float(ds[idx_best]),
                        "sampled_best_energy": sampled_best_energy,
                        "sampled_best_coords": sampled_best_coords,
                        "ds": ds,
                        "tp": tp,
                        "inds": inds,
                        "base_value": base_value
                    })
                else:
                    # no predictive candidate; but maybe sampled best improves
                    if current_energy - sampled_best_energy > ENERGY_ACCEPT_TOL:
                        candidates.append({
                            "idx": idx,
                            "which": "sampled",
                            "pred_disp": None,
                            "pred_energy_est": None,
                            "curvature": float(curvature) if curvature is not None else None,
                            "sampled_best_disp": float(ds[idx_best]),
                            "sampled_best_energy": sampled_best_energy,
                            "sampled_best_coords": sampled_best_coords,
                            "tp": tp,
                            "inds": inds,
                            "base_value": base_value
                        })

            # End processing internals -> candidates list ready
            if not candidates:
                if debug:
                    print("  No candidate predicted or sampled that improves energy on this subiteration.")
                break  # no candidates -> no more improvements this subiteration

            # PHASE 2: Validate top_k predicted candidates (by predicted deltaE), in parallel
            # Separate predicted candidates and sampled-only candidates; prefer predicted ones
            pred_candidates = [c for c in candidates if c['which'] == 'predicted']
            sampled_candidates = [c for c in candidates if c['which'] == 'sampled' or c['which'] == 'sampled_only']

            validated = []  # store dicts with true energy after validation
            # Prioritize predicted candidates by predicted improvement
            pred_candidates_sorted = sorted(pred_candidates, key=lambda c: (current_energy - c['pred_energy_est']), reverse=True)
            # pick top_k to validate (or fewer if not enough)
            to_validate = pred_candidates_sorted[:max(1, min(len(pred_candidates_sorted), top_k))]

            val_tasks = []
            for c in to_validate:
                # build coords_at_pred by applying predicted displacement to current_coords
                tp = c['tp']; inds = c['inds']; base_val = c['base_value']; d_pred = c['pred_disp']
                try:
                    if tp == 'bond':
                        i, j = inds
                        coords_at_pred = apply_bond_change(current_coords, i, j, base_val + d_pred, atoms)
                    elif tp == 'angle':
                        i, j, k = inds
                        coords_at_pred = apply_angle_change(current_coords, i, j, k, base_val + d_pred)
                    elif tp == 'dihedral':
                        i, j, k, l = inds
                        coords_at_pred = apply_dihedral_change(current_coords, i, j, k, l, base_val + d_pred)
                    else:
                        coords_at_pred = current_coords.copy()
                    xyz_pred = xyz_to_pyscf_string(atoms, coords_at_pred)
                    # submit validation as worker tasks
                    val_tasks.append((c['idx'], c['pred_disp'], xyz_pred, method, a_corr, b_hf, bs1, bs2, int(pyscf_threads_per_worker)))
                except Exception as e:
                    if debug:
                        print(f"    Failed to build predicted coords for candidate idx {c['idx']}: {e}")

            # run validation tasks in parallel (if any)
            def _val_worker(task):
                # task: (idx, pred_disp, xyz_pred, method, a_corr, b_hf, bs1, bs2, pyscf_threads)
                idx_local, pred_disp_local, xyz_pred_local, method_local, a_corr_local, b_hf_local, bs1_local, bs2_local, pthr = task
                try:
                    # set environment for linear algebra inside validation worker
                    os.environ['OMP_NUM_THREADS'] = str(pthr)
                    os.environ['MKL_NUM_THREADS'] = str(pthr)
                    os.environ['OPENBLAS_NUM_THREADS'] = str(pthr)
                except Exception:
                    pass
                try:
                    lib.num_threads(max(1, int(pthr)))
                except Exception:
                    pass
                try:
                    E_val = compute_cbs_energy_from_xyz_cached(xyz_pred_local, method_local, a_corr_local, b_hf_local, bs1_local, bs2_local)
                    return (idx_local, pred_disp_local, float(E_val), xyz_pred_local, None)
                except Exception as e:
                    return (idx_local, pred_disp_local, float('inf'), None, f"validation_error: {e}")

            val_results = []
            if val_tasks:
                # run validation tasks in the ProcessPoolExecutor as well
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
                    futures = {ex.submit(_val_worker, vt): vt for vt in val_tasks}
                    for fut in as_completed(futures):
                        val_results.append(fut.result())

            # fill validated list
            for vr in val_results:
                idx_local, pred_disp_local, E_val, xyz_pred_local, err = vr
                # find candidate metadata
                meta = next((c for c in pred_candidates_sorted if c['idx'] == idx_local and c['pred_disp'] == pred_disp_local), None)
                if meta is None:
                    continue
                deltaE_true = current_energy - float(E_val) if (E_val is not None and not math.isinf(E_val)) else -1.0
                validated.append({
                    "idx": idx_local,
                    "which": "predicted_validated",
                    "pred_disp": pred_disp_local,
                    "energy_validated": float(E_val) if (E_val is not None and not math.isinf(E_val)) else float('inf'),
                    "deltaE_true": float(deltaE_true),
                    "meta": meta,
                    "err": err,
                    "coords": None if xyz_pred_local is None else None  # we keep coords string in E_val step; we can recompute coords when applying
                })

            # After validating predicted candidates, if no validated candidate succeeded, fall back to sampled bests
            # Combine validated and sampled to choose the best to apply
            best_to_apply = None
            # pick best validated by deltaE_true
            if validated:
                # filter those with positive improvement above tolerance
                validated_good = [v for v in validated if v['deltaE_true'] > ENERGY_ACCEPT_TOL]
                if validated_good:
                    best_valid = max(validated_good, key=lambda v: v['deltaE_true'])
                    best_to_apply = {
                        "source": "predicted_validated",
                        "idx": best_valid['idx'],
                        "disp": best_valid['pred_disp'],
                        "energy_after": best_valid['energy_validated'],
                        "meta": best_valid['meta']
                    }

            # if no validated predicted candidate, consider sampled candidates (choose best sampled improvement)
            if best_to_apply is None and sampled_candidates:
                # choose sampled candidate with largest (current_energy - sampled_best_energy)
                sampled_good = [s for s in sampled_candidates if (current_energy - s['sampled_best_energy']) > ENERGY_ACCEPT_TOL]
                if sampled_good:
                    best_sampled = max(sampled_good, key=lambda s: (current_energy - s['sampled_best_energy']))
                    best_to_apply = {
                        "source": "sampled",
                        "idx": best_sampled['idx'],
                        "disp": best_sampled['sampled_best_disp'],
                        "energy_after": best_sampled['sampled_best_energy'],
                        "meta": best_sampled
                    }

            # If still nothing, break subiteration
            if best_to_apply is None:
                if debug:
                    print("  No validated predicted candidate and no sampled candidate with improvement; subiteration ends.")
                break

            # Apply best_to_apply: reconstruct coords for the chosen internal/displacement and update geometry
            chosen = best_to_apply
            meta = chosen['meta']
            tp = meta.get('tp') or meta.get('tp', None)
            inds = meta.get('inds')
            base_value = meta.get('base_value', meta.get('base_value', None))
            disp = chosen['disp']
            if tp is None:
                # fallback: try to get from internals
                icsrc = internals[chosen['idx']]
                tp = icsrc['type']; inds = icsrc['inds']; base_value = icsrc.get('value', _value_for_internal(tp, icsrc['inds'], current_coords))

            # Build coords after choosing this disp
            try:
                if tp == 'bond':
                    i, j = inds
                    new_coords = apply_bond_change(current_coords, i, j, base_value + disp, atoms)
                elif tp == 'angle':
                    i, j, k = inds
                    new_coords = apply_angle_change(current_coords, i, j, k, base_value + disp)
                elif tp == 'dihedral':
                    i, j, k, l = inds
                    new_coords = apply_dihedral_change(current_coords, i, j, k, l, base_value + disp)
                else:
                    new_coords = current_coords.copy()
            except Exception as e:
                if debug:
                    print(f"    Failed to construct coords to apply chosen candidate: {e}")
                # remove this candidate and continue subiteration
                # (prevent infinite loop)
                if chosen['source'] == 'predicted_validated':
                    # remove this idx from pred_candidates_sorted for next subiteration
                    pred_candidates_sorted = [c for c in pred_candidates_sorted if c['idx'] != chosen['idx']]
                else:
                    sampled_candidates = [s for s in sampled_candidates if s['idx'] != chosen['idx']]
                continue

            # Re-evaluate energy at new_coords to be sure (and to have canonical energy)
            try:
                xyz_after = xyz_to_pyscf_string(atoms, new_coords)
                E_after = compute_cbs_energy_from_xyz_cached(xyz_after, method, a_corr, b_hf, bs1, bs2)
            except Exception as e:
                if debug:
                    print(f"    Evaluation after applying candidate failed: {e}")
                # reject and continue
                if chosen['source'] == 'predicted_validated':
                    pred_candidates_sorted = [c for c in pred_candidates_sorted if c['idx'] != chosen['idx']]
                else:
                    sampled_candidates = [s for s in sampled_candidates if s['idx'] != chosen['idx']]
                continue

            # Accept the change
            applied_changes.append({
                "internal": _label_internal(tp, inds, atoms),
                "source": chosen['source'],
                "disp": float(disp),
                "energy_after": float(E_after)
            })
            current_coords = new_coords.copy()
            # rebuild internals on the updated geometry
            internals = generate_internals_from_geometry(atoms, current_coords)
            # update current energy
            current_energy = float(E_after)

            if debug:
                print(f"  Applied change: {applied_changes[-1]}")

            # After applying, continue subiteration (re-scan all internals on updated geometry)
            # (this 'while True' loop will repeat)
        # end while subiteration

        # record cycle energy and internals trace
        history.append({'cycle': cycle, 'energy': float(current_energy)})
        cyc_map = {}
        # compute baseline labels values (canonical baseline internals from start)
        for lbl, tp, inds in zip(baseline_labels, baseline_types, baseline_inds):
            try:
                cyc_map[lbl] = _value_for_internal(tp, inds, current_coords)
            except Exception:
                cyc_map[lbl] = None
        internals_trace.append(cyc_map)

        # per-cycle debug summary
        if applied_changes:
            print(f"\nCycle {cycle} applied changes ({len(applied_changes)}):")
            for ch in applied_changes:
                print(f"  - {ch['internal']}: source={ch['source']}, disp={ch['disp']:.6f}, E_after={ch['energy_after']:.10f}")
        else:
            print(f"Cycle {cycle}: no accepted internal changes.")

        # convergence check
        if cycle > 1:
            ediff = abs(history[-1]['energy'] - history[-2]['energy'])
            print(f"  ΔE since last cycle: {ediff:.4e} Ha")
            if ediff < ENERGY_CRIT:
                print("Converged by energy criterion")
                converged = True
                break

        displacement_factor *= CUT

    return atoms, current_coords, history, converged, baseline_labels, internals_trace


# -------------------------
# run_optimization API (CLI wrapper)
# -------------------------
def run_optimization(params: dict, outputs_dir: Path):
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
    basis1 = params.get("basis1")
    basis2 = params.get("basis2")
    basis_pair = (basis1, basis2) if (basis1 and basis2) else None
    # parallelism params
    workers = int(params.get("workers", DEFAULT_WORKERS))
    top_k = int(params.get("top_k", DEFAULT_TOP_K))
    pyscf_threads_per_worker = int(params.get("pyscf_threads_per_worker", 1))
    debug = bool(params.get("debug", False))

    atoms, coords0, comment = read_xyz(input_xyz)
    print(f"Loaded {len(atoms)} atoms from {input_xyz}")
    print(f"Parallel SQM: workers={workers}, top_k={top_k}, pyscf_threads_per_worker={pyscf_threads_per_worker}")
    print("Building redundant internal coordinates (initial)...")
    internals = generate_internals_from_geometry(atoms, coords0)
    print(f"Found {len(internals)} internal coordinates (bonds/angles/dihedrals).")

    atoms_out, coords_out, history, converged, baseline_labels, internals_trace = optimize_from_xyz(
        atoms,
        coords0,
        method=method,
        maxcycle=maxcycle,
        fac_mult=fac,
        x1=x1, x2=x2, x1_hf=x1hf, x2_hf=x2hf, beta=beta,
        basis_pair=basis_pair,
        workers=workers,
        top_k=top_k,
        pyscf_threads_per_worker=pyscf_threads_per_worker,
        debug=debug
    )

    base = Path(input_xyz).stem
    prefix = f"{base}_SQM"
    cycles_file = write_cycle_energies(outputs_dir, prefix, history)
    xyz_file = write_final_xyz(outputs_dir, prefix, atoms_out, coords_out, history[-1]['energy'] if history else 0.0)

    try:
        print("\nRedundant-internal coordinates trace (rows: internal, columns: cycle 0..):")
        ncycles = len(internals_trace)
        header = ["INTERNAL"] + [f"C{idx}" for idx in range(0, ncycles)]
        print("  ".join(header))
        for lbl in baseline_labels:
            row_vals = [str(internals_trace[c].get(lbl, "-")) for c in range(0, ncycles)]
            print(lbl + "  " + "  ".join(row_vals))
    except Exception as e:
        print("Failed to print internals trace table:", e)

    return {
        "history": history,
        "final_energy": float(history[-1]['energy']) if history else None,
        "final_cart": coords_out,
        "symbols": atoms_out,
        "outputs": {"cycles": str(cycles_file), "xyz": str(xyz_file)},
        "converged": converged,
        "internals_trace": {"labels": baseline_labels, "trace": internals_trace},
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
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of worker processes")
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of predicted minima to validate per subiteration")
    p.add_argument("--pyscf_threads_per_worker", type=int, default=1, help="Threads per worker (set 1 for stability)")
    p.add_argument("--debug", action="store_true", help="Enable verbose diagnostics")
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
        "workers": args.workers,
        "top_k": args.top_k,
        "pyscf_threads_per_worker": args.pyscf_threads_per_worker,
        "debug": args.debug
    }
    out = Path(args.out)
    result = run_optimization(params, out)
    print("Done. Outputs written to:", out)