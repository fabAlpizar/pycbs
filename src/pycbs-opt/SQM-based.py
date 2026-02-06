#!/usr/bin/env python3
"""
SQM-based optimizer module (no geomeTRIC; robust RIC manager + 3-point fit)

Features:
 - Robust geometry-derived redundant internal coordinates (bonds / angles / dihedrals).
 - 3-point parabolic fit per-internal (points: -h, 0, +h).
 - Best-improvement subiterations: evaluate all internals, apply only the single best validated improvement,
   and repeat until no acceptable candidate remains.
 - Validation: predicted parabolic minimum is converted to Cartesian, its true CBS energy is computed,
   and acceptance is only performed if the **evaluated** energy at that geometry improves sufficiently.
 - ENERGY_ACCEPT_TOL and MIN_CURVATURE thresholds to reject tiny/noisy moves.
 - Per-cycle debug summary printing all applied changes and final status.
 - Compatible with your existing `pycbs.writer.write_cycle_energies` and `write_final_xyz`.
"""

from pathlib import Path
import math
import sys
import traceback
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
basis_sets = ['cc-pvdz', 'cc-pvtz']
DEFAULT_METHOD = 'CCSD(T)'

X1_DEFAULT, X2_DEFAULT = 1.85, 2.639
X1HF_DEFAULT, X2HF_DEFAULT = 3.02, 3.64
BETA_DEFAULT = 1.62

PYSCF_THREADS = 6
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
    # use 1-based indices for readability, include atom symbols
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
    """
    Build a redundant internal set from geometry:
      - bonds: distance <= scale * (rcov_i + rcov_j)
      - angles: i-j-k where i and k bonded to j (i < k to avoid duplicates)
      - dihedrals: i-j-k-l where chain of bonds exists
    Returns list of dicts {'type','inds','value'} with zero-based indices.
    """
    nat = len(atoms)
    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    bonds = set()
    for i in range(nat):
        for j in range(i + 1, nat):
            ri = _covalent_radius(atoms[i])
            rj = _covalent_radius(atoms[j])
            cutoff = scale * (ri + rj)
            # ignore extremely short distances (< 0.2 Å) as suspect
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
@lru_cache(maxsize=2000)
def _compute_cbs_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str, spin: int):
    """
    Calculate CBS energy using two basis sets (bs1, bs2) and extrapolation coefficients
    a_corr, b_hf. Raises on SCF failure (so the caller treats that displacement as invalid).
    """
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

        # check convergence
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
            # final fallback to MP2 if CCSD(T) failed
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
    # thin wrapper for readability
    return _compute_cbs_from_xyz_cached(xyz_string, method, a_corr, b_hf, bs1, bs2, int(spin))


# -------------------------
# 3-point parabolic helper (robust)
# -------------------------
def parabolic_minimum_3pt(x, y):
    """
    Fit quadratic using three points (x0,x1,x2) and return x_min, y_min.
    If fit fails, return the sampled best.
    """
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
# Main optimization loop (SQM-style) — no geomeTRIC
# -------------------------
def optimize_from_xyz(atoms, coords, method=DEFAULT_METHOD, maxcycle=MAXCYCLE_DEFAULT, fac_mult=FAC_DEFAULT,
                      x1=X1_DEFAULT, x2=X2_DEFAULT, x1_hf=X1HF_DEFAULT, x2_hf=X2HF_DEFAULT, beta=BETA_DEFAULT,
                      basis_pair=None, spin: int = 0, debug: bool = False, energy_accept_tol: float | None = None):
    """
    SQM-style optimizer using geometry-only RIC manager and 3-point fitting.

    Returns atoms, final_coords, history(list of dicts), converged(bool), baseline_labels, internals_trace
    """
    # acceptance / diagnostics thresholds (local defaults)
    ENERGY_ACCEPT_TOL = 1e-6 if energy_accept_tol is None else float(energy_accept_tol)  # Ha
    MIN_CURVATURE = 1e-10     # very small positive floor to avoid numerical instabilities

    a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
    denom = math.exp(beta * x2_hf) - math.exp(beta * x1_hf)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    b_hf = math.exp(beta * x1_hf) / denom

    if basis_pair is None:
        basis_pair = (basis_sets[0], basis_sets[1])
    bs1, bs2 = basis_pair

    # Build baseline internals (canonical labels)
    baseline_internals = generate_internals_from_geometry(atoms, coords)
    if not baseline_internals:
        raise RuntimeError("No internal coordinates found; check input geometry.")

    baseline_labels = [_label_internal(ic['type'], ic['inds'], atoms) for ic in baseline_internals]
    baseline_inds = [ic['inds'] for ic in baseline_internals]
    baseline_types = [ic['type'] for ic in baseline_internals]

    # Working internals (will be rebuilt after geometry changes)
    internals = [dict(ic) for ic in baseline_internals]

    current_coords = coords.copy()
    displacement_factor = fac_mult
    history = []
    converged = False

    internals_trace = []
    # cycle 0 initial values
    init_vals = {}
    for lbl, tp, inds in zip(baseline_labels, baseline_types, baseline_inds):
        init_vals[lbl] = _value_for_internal(tp, inds, current_coords)
    internals_trace.append(init_vals)

    try:
        _compute_cbs_from_xyz_cached.cache_clear()
    except Exception:
        pass

    for cycle in range(1, maxcycle + 1):
        # compute current energy from the actual geometry
        cur_xyz = xyz_to_pyscf_string(atoms, current_coords)
        try:
            current_energy = compute_cbs_energy_from_xyz_cached(cur_xyz, method, a_corr, b_hf, bs1, bs2, spin)
        except Exception as e:
            raise RuntimeError(f"CBS evaluation at cycle start failed: {e}")

        applied_changes = []  # collect per-cycle change summaries for debug printing

        if debug:
            print(f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}, current E = {current_energy:.10f} Ha")
        else:
            print(f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}, current E = {current_energy:.10f} Ha")

        # Best-improvement subiteration loop
        while True:
            best_candidate = None  # (deltaE, predicted_xmin, coords_at_xmin, ic, curvature, grad0, sampled_best_val, sampled_best_coords)
            for ic in internals:
                tp = ic['type']; inds = ic['inds']
                # displacement magnitude for 3-point: ±h around current internal value
                if tp == 'bond':
                    base = ic.get('value', _value_for_internal(tp, inds, current_coords))
                    h = displacement_factor * base
                    ds = np.array([-h, 0.0, h], dtype=float)
                elif tp in ('angle', 'dihedral'):
                    base = ic.get('value', _value_for_internal(tp, inds, current_coords))
                    # angle step uses degrees; scale displacement_factor up as before
                    h = 2.0 * (displacement_factor * 100.0)  # this mirrors previous magnitude heuristic
                    ds = np.array([-h, 0.0, h], dtype=float)
                else:
                    ds = np.array([0.0])
                    h = 0.0

                es = []
                coords_list = []
                scf_failed_any = False
                for d in ds:
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
                        es.append(float(E)); coords_list.append(new_coords)
                    except Exception as exc:
                        es.append(float('inf')); coords_list.append(None)
                        scf_failed_any = True
                        if debug:
                            print(f"    eval failed for IC {tp} {inds} displacement {d}: {exc}")

                es = np.array(es, dtype=float)
                if np.all(np.isinf(es)):
                    if debug:
                        print(f"  All evaluations failed for IC {tp} {inds}; skipping")
                    continue

                # parabolic fit using three points
                x_min_disp, e_min = parabolic_minimum_3pt(ds, es)
                idx_best = int(np.nanargmin(es))
                sampled_best_coords = coords_list[idx_best]
                sampled_best_energy = float(es[idx_best]) if not np.isinf(es[idx_best]) else None

                # numeric gradient and curvature (central finite difference with 3 points)
                grad0 = None
                curvature = None
                try:
                    Eminus = es[0]; E0 = es[1]; Eplus = es[2]
                    grad0 = (Eplus - Eminus) / (2.0 * (ds[2] - ds[1]))  # central slope
                    curvature = (Eplus + Eminus - 2.0 * E0) / ((ds[2] - ds[1]) ** 2)
                except Exception:
                    grad0 = None
                    curvature = None

                # candidate improvement relative to current_energy
                deltaE_pred = current_energy - e_min  # positive => predicted improvement

                if debug:
                    cur_val = ic.get('value', _value_for_internal(tp, inds, current_coords))
                    pred_val = cur_val + x_min_disp
                    sampled_best_val = None
                    try:
                        sampled_best_val = _value_for_internal(tp, inds, sampled_best_coords) if sampled_best_coords is not None else None
                    except Exception:
                        sampled_best_val = None
                    print(f"  IC {tp} {inds}: cur={cur_val:.6f}, ds=[{ds[0]:.4g}, 0, {ds[2]:.4g}], "
                          f"E_samples=[{es[0]:.6f}, {es[1]:.6f}, {es[2]:.6f}], x_min_disp={x_min_disp:.6f}, e_min={e_min:.10f}")
                    if grad0 is not None:
                        print(f"    numeric grad @0 = {grad0:.3e}, curvature = {curvature:.3e}")
                    if sampled_best_val is not None:
                        print(f"    predicted internal (pred_val) = {pred_val:.6f}, best sampled internal = {sampled_best_val:.6f}")
                    if scf_failed_any:
                        print("    NOTE: some SCF evaluations failed for this IC sampling")

                # Selection criteria: require convex curvature, predicted improvement above tolerance
                if curvature is not None and curvature > MIN_CURVATURE and deltaE_pred > ENERGY_ACCEPT_TOL:
                    # We will attempt to validate — produce coordinates at predicted minimum (if inside bracket)
                    # Only accept if validation evaluation yields improvement (evaluate true energy at predicted coords)
                    # Build coords at predicted minimum
                    coords_at_pred = None
                    try:
                        # produce coords applying predicted displacement
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

                        xyz_pred = xyz_to_pyscf_string(atoms, coords_at_pred)
                        E_pred_true = compute_cbs_energy_from_xyz_cached(xyz_pred, method, a_corr, b_hf, bs1, bs2, spin)
                        deltaE_pred_true = current_energy - float(E_pred_true)
                    except Exception:
                        # predicted geometry evaluation failed (SCF or correlation); fall back to sampled best candidate instead
                        E_pred_true = float('inf')
                        deltaE_pred_true = -1.0

                    # If predicted true energy is acceptable, candidate becomes this predicted geometry
                    if E_pred_true != float('inf') and deltaE_pred_true > ENERGY_ACCEPT_TOL:
                        # choose predicted coords as candidate
                        if best_candidate is None or deltaE_pred_true > best_candidate[0]:
                            best_candidate = (deltaE_pred_true, x_min_disp, coords_at_pred, ic, curvature, grad0, sampled_best_energy, sampled_best_coords, 'predicted')
                    else:
                        # fallback: consider sampled best point (must also improve actual energy)
                        if sampled_best_energy is not None and (current_energy - sampled_best_energy) > ENERGY_ACCEPT_TOL:
                            if best_candidate is None or (current_energy - sampled_best_energy) > best_candidate[0]:
                                best_candidate = ((current_energy - sampled_best_energy), ds[idx_best], sampled_best_coords, ic, curvature, grad0, sampled_best_energy, sampled_best_coords, 'sampled')

            # end for internals

            if best_candidate is None:
                if debug:
                    print("  No acceptable candidate found in this subiteration.")
                break

            # Apply best candidate (either predicted-validated or sampled)
            deltaE, chosen_disp, chosen_coords, chosen_ic, curvature, grad0, sampled_e, sampled_coords, which = best_candidate
            # Apply
            current_coords = chosen_coords.copy()
            # Refresh working internals from geometry
            internals = generate_internals_from_geometry(atoms, current_coords)
            # Update current_energy to the **actual evaluated** energy at the applied coordinates.
            # If we applied a predicted candidate we evaluated it already (E_pred_true); if sampled we have sampled energy.
            try:
                cur_xyz_after = xyz_to_pyscf_string(atoms, current_coords)
                actual_E_after = compute_cbs_energy_from_xyz_cached(cur_xyz_after, method, a_corr, b_hf, bs1, bs2, spin)
            except Exception:
                # this should not normally happen because we validated it, but be safe
                actual_E_after = float('inf')

            # Record applied change
            applied_changes.append({
                "internal": _label_internal(chosen_ic['type'], chosen_ic['inds'], atoms),
                "which": which,
                "disp": float(chosen_disp),
                "deltaE_estimate": float(deltaE),
                "energy_after": float(actual_E_after),
                "curvature": float(curvature) if curvature is not None else None,
                "grad0": float(grad0) if grad0 is not None else None
            })

            # Update current energy to the true evaluated energy
            current_energy = float(actual_E_after)

            # Continue subiteration searching for another best internal on the updated geometry
            # (this loop will re-evaluate all internals on the new geometry)
        # end while subiteration

        # record cycle energy and internals trace
        history.append({'cycle': cycle, 'energy': float(current_energy)})
        cyc_map = {}
        for lbl, tp, inds in zip(baseline_labels, baseline_types, baseline_inds):
            cyc_map[lbl] = _value_for_internal(tp, inds, current_coords)
        internals_trace.append(cyc_map)

        # per-cycle debug summary: list all applied changes this cycle
        if applied_changes:
            print(f"\nCycle {cycle} applied changes ({len(applied_changes)}):")
            for ch in applied_changes:
                print(f"  - {ch['internal']}: type={ch['which']}, disp={ch['disp']:.6f}, ΔE_est={ch['deltaE_estimate']:.3e} Ha, E_after={ch['energy_after']:.10f}, curvature={ch['curvature']:.3e}")
        else:
            print(f"Cycle {cycle}: no accepted internal changes.")

        # convergence check (energy)
        if cycle > 1:
            ediff = abs(history[-1]['energy'] - history[-2]['energy'])
            print(f"  ΔE since last cycle: {ediff:.4e} Ha")
            if ediff < ENERGY_CRIT:
                print("Converged by energy criterion")
                converged = True
                break

        displacement_factor *= CUT

    # return canonical baseline_labels and internals_trace for reproducibility
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
        energy_accept_tol=energy_accept_tol
    )

    base = Path(input_xyz).stem
    prefix = f"{base}_SQM"
    cycles_file = write_cycle_energies(outputs_dir, prefix, history)
    xyz_file = write_final_xyz(outputs_dir, prefix, atoms_out, coords_out, history[-1]['energy'] if history else 0.0)

    # Print internals table (raw floats)
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
        "energy_accept_tol": args.energy_accept_tol
    }
    out = Path(args.out)
    result = run_optimization(params, out)
    print("Done. Outputs written to:", out)