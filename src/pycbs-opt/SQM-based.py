#!/usr/bin/env python3
"""
SQM-based optimizer module (improved + immediate 5-point exploration per-internal)

- Preserves improvements from your "improved" script:
    * larger LRU cache for CBS evaluations
    * geometry helpers, atom masses/radii
    * diagnostic printing and internals trace
    * ability to parallelize per-displacement evaluations (workers)
- Changes exploration to match the "full" script:
    * for each internal perform a 5-point scan (−2h, −h, 0, +h, +2h or angle/dihedral analog)
    * if the best sampled/fitted geometry for that internal improves the current energy
      by more than ENERGY_ACCEPT_TOL, **apply the update immediately** and continue with the next internal
Usage:
    python sqm_optimizer_immediate5_improved.py -i input.xyz --out PyCBS-OUTPUTS --workers 1 --debug
"""
from pathlib import Path
import math
import sys
import traceback
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

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
basis_sets = ['cc-pvdz', 'cc-pvtz']
DEFAULT_METHOD = 'CCSD(T)'

X1_DEFAULT, X2_DEFAULT = 1.85, 2.639
X1HF_DEFAULT, X2HF_DEFAULT = 3.02, 3.64
BETA_DEFAULT = 1.62

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

# Covalent radii (Å) - used by geometry fallback
COVALENT_RADII = {
    'H': 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
    'K': 2.03, 'Ca': 1.76
}
DEFAULT_COV_RAD = 0.77  # fallback if element not in dict

DEFAULT_SPIN = 0

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
    # minimized string work
    return "\n".join(f"{a} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}" for a, c in zip(atoms, coords))


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
    # bonds
    for i, j in bonds:
        out.append({"type": "bond", "inds": (i, j), "value": float(dmat[i, j])})

    # angles
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

    # dihedrals
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
# bigger cache for more reuse
@lru_cache(maxsize=10000)
def _compute_cbs_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str, spin: int):
    scf_vals = []
    corr_vals = []
    for basis in (bs1, bs2):
        mol = gto.Mole()
        mol.atom = xyz_string
        mol.basis = basis
        mol.spin = int(spin)
        mol.charge = 0
        mol.nthread = PYSCF_THREADS
        mol.max_memory = 8000
        mol.build()

        mf = scf.RHF(mol)
        mf.max_memory = 14330
        mf.conv_tol = 1e-9
        mf.max_cycle = 100
        scf_energy = mf.kernel()

        scf_converged = getattr(mf, "converged", True)
        if not scf_converged:
            raise RuntimeError("SCF failed to converge for this geometry (mf.converged is False)")

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


def compute_cbs_energy_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str, spin: int):
    return _compute_cbs_from_xyz_cached(xyz_string, method, a_corr, b_hf, bs1, bs2, int(spin))


# -------------------------
# 3/5-point parabolic helpers
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
# Main optimization loop (immediate 5-point exploration per internal)
# -------------------------
def optimize_from_xyz(atoms, coords, method=DEFAULT_METHOD, maxcycle=MAXCYCLE_DEFAULT, fac_mult=FAC_DEFAULT,
                      x1=X1_DEFAULT, x2=X2_DEFAULT, x1_hf=X1HF_DEFAULT, x2_hf=X2HF_DEFAULT, beta=BETA_DEFAULT,
                      basis_pair=None, spin: int = 0, debug: bool = False, energy_accept_tol: float | None = None,
                      workers: int = 1):
    ENERGY_ACCEPT_TOL = 1e-6 if energy_accept_tol is None else float(energy_accept_tol)  # Ha
    MIN_CURVATURE = 1e-10

    a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
    denom = math.exp(beta * x2_hf) - math.exp(beta * x1_hf)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    b_hf = math.exp(beta * x1_hf) / denom

    if basis_pair is None:
        basis_pair = (basis_sets[0], basis_sets[1])
    bs1, bs2 = basis_pair

    internals = generate_internals_from_geometry(atoms, coords)
    if not internals:
        raise RuntimeError("No internal coordinates found; check input geometry.")

    baseline_labels = [_label_internal(ic['type'], ic['inds'], atoms) for ic in internals]
    baseline_inds = [ic['inds'] for ic in internals]
    baseline_types = [ic['type'] for ic in internals]

    current_coords = coords.copy()
    displacement_factor = fac_mult
    history = []
    converged = False

    internals_trace = []
    init_vals = {}
    for lbl, tp, inds in zip(baseline_labels, baseline_types, baseline_inds):
        init_vals[lbl] = _value_for_internal(tp, inds, current_coords)
    internals_trace.append(init_vals)

    try:
        _compute_cbs_from_xyz_cached.cache_clear()
    except Exception:
        pass

    for cycle in range(1, maxcycle + 1):
        cur_xyz = xyz_to_pyscf_string(atoms, current_coords)  # compute once per cycle
        try:
            current_energy = compute_cbs_energy_from_xyz_cached(cur_xyz, method, a_corr, b_hf, bs1, bs2, spin)
        except Exception as e:
            raise RuntimeError(f"CBS evaluation at cycle start failed: {e}")

        applied_changes = []

        print(f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}, current E = {current_energy:.10f} Ha")

        # iterate internals sequentially, 5-point scan per internal, apply immediate update if improvement
        for ic in list(internals):  # iterate over snapshot of internals
            tp = ic['type']
            inds = ic['inds']
            if tp == 'bond':
                base = ic.get('value', _value_for_internal(tp, inds, current_coords))
                h = displacement_factor * base
                ds = np.array([-h, 0.0, +h], dtype=float)

            elif tp in ('angle', 'dihedral'):
                base = ic.get('value', _value_for_internal(tp, inds, current_coords))
                ddeg = 2.0 * (displacement_factor * 100.0)
                ds = np.array([-ddeg, 0.0, +ddeg], dtype=float)

            energies = [float('inf')] * ds.size
            coords_list = [None] * ds.size

            # helper to evaluate one displacement
            def eval_disp(idx, d):
                try:
                    if tp == 'bond':
                        i, j = inds
                        new_val = ic.get('value', _value_for_internal(tp, inds, current_coords)) + d
                        new_coords = apply_bond_change(current_coords, i, j, new_val, atoms)
                    elif tp == 'angle':
                        i, j, k = inds
                        new_val = ic.get('value', _value_for_internal(tp, inds, current_coords)) + d
                        new_coords = apply_angle_change(current_coords, i, j, k, new_val)
                    elif tp == 'dihedral':
                        i, j, k, l = inds
                        new_val = ic.get('value', _value_for_internal(tp, inds, current_coords)) + d
                        new_coords = apply_dihedral_change(current_coords, i, j, k, l, new_val)
                    else:
                        new_coords = current_coords.copy()
                    xyzs = xyz_to_pyscf_string(atoms, new_coords)
                    E = compute_cbs_energy_from_xyz_cached(xyzs, method, a_corr, b_hf, bs1, bs2, spin)
                    return float(E), new_coords
                except Exception as e:
                    if debug:
                        print(f"    eval failed for IC {tp} {inds} displacement {d}: {e}")
                    return float('inf'), None

            if workers is None or workers <= 1:
                for idx, d in enumerate(ds):
                    energies[idx], coords_list[idx] = eval_disp(idx, d)
            else:
                # parallelize the 5 displacements for this single internal
                with ThreadPoolExecutor(max_workers=workers) as exc:
                    fut_map = {exc.submit(eval_disp, idx, float(d)): idx for idx, d in enumerate(ds)}
                    for fut in as_completed(fut_map):
                        idx = fut_map[fut]
                        try:
                            E, new_coords = fut.result()
                            energies[idx] = E
                            coords_list[idx] = new_coords
                        except Exception as e:
                            energies[idx] = float('inf')
                            coords_list[idx] = None
                            if debug:
                                print("    worker error:", e)

            es = np.array(energies, dtype=float)
            if np.all(np.isinf(es)):
                if debug:
                    print(f"  Skipping {tp} {inds}: all evals failed")
                continue

            # Evaluate parabolic min across the grid (5-point) — returns x_min (in displacement units) and e_min
            try:
                x_min_disp, e_min = parabolic_minimum(ds, es)
            except Exception:
                # fallback
                idx_best = int(np.nanargmin(es))
                x_min_disp = float(ds[idx_best])
                e_min = float(es[idx_best])

            idx_sampled_best = int(np.nanargmin(es))
            sampled_best_coords = coords_list[idx_sampled_best]
            sampled_best_energy = float(es[idx_sampled_best]) if not np.isinf(es[idx_sampled_best]) else None

            # compute curvature and grad estimate (from central three points if available)
            grad0 = None
            curvature = None
            try:
                # center three points are indices 1,2,3 for 5-point array
                Eminus = es[1]; E0 = es[2]; Eplus = es[3]
                grad0 = (Eplus - Eminus) / (2.0 * (ds[3] - ds[2]))
                curvature = (Eplus + Eminus - 2.0 * E0) / ((ds[3] - ds[2]) ** 2)
            except Exception:
                pass

            # predicted displacement geometry (via parabolic fit) -> construct coordinates_at_pred and evaluate true energy
            coords_at_pred = None
            E_pred_true = float('inf')
            deltaE_pred_true = -1.0
            try:
                if tp == 'bond':
                    i, j = inds
                    coords_at_pred = apply_bond_change(current_coords, i, j, ic.get('value', _value_for_internal(tp, inds, current_coords)) + x_min_disp, atoms)
                elif tp == 'angle':
                    i, j, k = inds
                    coords_at_pred = apply_angle_change(current_coords, i, j, k, ic.get('value', _value_for_internal(tp, inds, current_coords)) + x_min_disp)
                elif tp == 'dihedral':
                    i, j, k, l = inds
                    coords_at_pred = apply_dihedral_change(current_coords, i, j, k, l, ic.get('value', _value_for_internal(tp, inds, current_coords)) + x_min_disp)
                else:
                    coords_at_pred = current_coords.copy()
                if coords_at_pred is not None:
                    xyz_pred = xyz_to_pyscf_string(atoms, coords_at_pred)
                    E_pred_true = compute_cbs_energy_from_xyz_cached(xyz_pred, method, a_corr, b_hf, bs1, bs2, spin)
                    deltaE_pred_true = current_energy - float(E_pred_true)
            except Exception:
                E_pred_true = float('inf')
                deltaE_pred_true = -1.0

            # Decision logic: accept predicted parabolic minimum if validated, else accept sampled best if it improves
            accepted = False
            accept_reason = None
            chosen_coords = None
            chosen_disp = None
            chosen_energy_after = None

            if E_pred_true != float('inf') and deltaE_pred_true > ENERGY_ACCEPT_TOL and curvature is not None and curvature > MIN_CURVATURE:
                # accept predicted
                accepted = True
                accept_reason = 'predicted'
                chosen_coords = coords_at_pred
                chosen_disp = float(x_min_disp)
                chosen_energy_after = float(E_pred_true)
            else:
                # fallback to sampled best
                if sampled_best_coords is not None and sampled_best_energy is not None and (current_energy - sampled_best_energy) > ENERGY_ACCEPT_TOL:
                    accepted = True
                    accept_reason = 'sampled'
                    chosen_coords = sampled_best_coords
                    chosen_disp = float(ds[idx_sampled_best])
                    chosen_energy_after = float(sampled_best_energy)

            if accepted:
                # apply update immediately (match second script behavior)
                prev_energy = current_energy
                current_coords = chosen_coords.copy()
                # rebuild internals on the *new* geometry
                try:
                    internals = generate_internals_from_geometry(atoms, current_coords)
                except Exception:
                    # if rebuild fails keep previous internals but continue
                    pass

                # try to compute actual energy after update (validate and store)
                try:
                    cur_xyz_after = xyz_to_pyscf_string(atoms, current_coords)
                    actual_E_after = compute_cbs_energy_from_xyz_cached(cur_xyz_after, method, a_corr, b_hf, bs1, bs2, spin)
                except Exception:
                    actual_E_after = float('inf')

                applied_changes.append({
                    "internal": _label_internal(tp, inds, atoms),
                    "which": accept_reason,
                    "disp": float(chosen_disp),
                    "deltaE_estimate": float(prev_energy - chosen_energy_after) if chosen_energy_after is not None else None,
                    "energy_after": float(actual_E_after) if actual_E_after != float('inf') else None,
                    "curvature": float(curvature) if curvature is not None else None,
                    "grad0": float(grad0) if grad0 is not None else None
                })

                # update current_energy to the actual energy after change if available
                if actual_E_after != float('inf'):
                    current_energy = float(actual_E_after)
                else:
                    # fallback to the sampled/predicted estimate
                    if chosen_energy_after is not None:
                        current_energy = float(chosen_energy_after)

                # After immediate update we continue to the next internal (using the new current_coords)
                if debug:
                    print(f"  Applied update on {tp} {inds}: reason={accept_reason}, disp={chosen_disp:.6f}, E_after={current_energy:.10f}")

            else:
                if debug:
                    print(f"  IC {tp} {inds}: no accepted improvement (best E {float(np.nanmin(es)):.10f})")

        # end per-internal loop for this cycle

        history.append({'cycle': cycle, 'energy': float(current_energy)})
        cyc_map = {}
        for lbl, tp, inds in zip(baseline_labels, baseline_types, baseline_inds):
            cyc_map[lbl] = _value_for_internal(tp, inds, current_coords)
        internals_trace.append(cyc_map)

        if applied_changes:
            print(f"\nCycle {cycle} applied changes ({len(applied_changes)}):")
            for ch in applied_changes:
                print(f"  - {ch['internal']}: type={ch['which']}, disp={ch['disp']:.6f}, ΔE_est={ch['deltaE_estimate']:.3e} Ha, E_after={ch['energy_after']:.10f}, curvature={ch['curvature']}")
        else:
            print(f"Cycle {cycle}: no accepted internal changes.")

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
    spin = int(params.get("spin", DEFAULT_SPIN))
    debug = bool(params.get("debug", False))
    energy_accept_tol = params.get("energy_accept_tol", None)
    workers = int(params.get("workers", 1))

    atoms, coords0, comment = read_xyz(input_xyz)
    print(f"Loaded {len(atoms)} atoms from {input_xyz}")
    print("Building redundant internal coordinates (geometry-derived fallback)...")
    internals = generate_internals_from_geometry(atoms, coords0)
    print(f"Found {len(internals)} internals.")

    atoms_out, coords_out, history, converged, baseline_labels, internals_trace = optimize_from_xyz(
        atoms,
        coords0,
        method=method,
        maxcycle=maxcycle,
        fac_mult=fac,
        x1=x1, x2=x2, x1_hf=x1hf, x2_hf=x2hf, beta=beta,
        basis_pair=basis_pair,
        spin=spin,
        debug=debug,
        energy_accept_tol=energy_accept_tol,
        workers=workers
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
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--debug", action="store_true", help="Enable verbose diagnostics")
    p.add_argument("--energy_accept_tol", type=float, default=None, help="Per-move acceptance energy (Ha)")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel workers for per-internal displacement evaluations")
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
        "spin": args.spin,
        "debug": args.debug,
        "energy_accept_tol": args.energy_accept_tol,
        "workers": args.workers
    }
    out = Path(args.out)
    result = run_optimization(params, out)
    print("Done. Outputs written to:", out)