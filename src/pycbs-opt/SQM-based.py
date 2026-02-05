#!/usr/bin/env python3
"""
SQM-based optimizer module (full implementation)

This implements the "SQM-style" redundant-internal-coordinate optimizer:
- reads an input XYZ
- builds a redundant set of internals (bonds, angles, dihedrals)
- for each internal coordinate performs a 5-point finite-difference
  (parabolic) scan and tries to update geometry if the CBS energy improves
- repeats for a number of cycles or until convergence

Changes in this variant:
- When available, attempts to use geomeTRIC to build a robust set of
  redundant internal coordinates (best-effort; falls back to local builder
  if geomeTRIC isn't available or integration fails).
- Keeps a baseline list of redundant internals (labels) and records the
  value of each baseline internal at every cycle for reproducibility.
- At the end of an optimization prints a table to the terminal: rows are
  the redundant internals (labeled) and columns are the value of that
  internal for each cycle (raw float strings).
- Existing behavior preserved (spin propagation, caching, filename prefixing).
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

# Try to import geomeTRIC (best-effort). Integration is optional and falls back to
# the internal builder if geomeTRIC isn't present or usable.
_GEOMETRIC = None
try:
    # try common import names (some installations expose 'geometric' or 'geomeTRIC')
    try:
        import geometric as _geometric_mod  # type: ignore
        _GEOMETRIC = _geometric_mod
    except Exception:
        import geomeTRIC as _geometric_mod  # type: ignore
        _GEOMETRIC = _geometric_mod
except Exception:
    _GEOMETRIC = None

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
MAXCYCLE_DEFAULT = 50
ENERGY_CRIT = 1e-8
FAC_DEFAULT = 0.05
CUT = 0.75

# covalent radii / masses (used by fallback builder)
COV_RAD = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02
}
ATOMIC_MASS = {
    'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.999, 'F': 18.998,
    'P': 30.9738, 'S': 32.065, 'Cl': 35.453
}

# module-level default spin (can be set from run_optimization params)
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


def distance(a, b):
    return np.linalg.norm(a - b)


def _angle_deg(a, b, c):
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-16)
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _dihedral_deg(p0, p1, p2, p3):
    # return dihedral in degrees
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= (np.linalg.norm(b1) + 1e-16)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


# -------------------------
# Internals: try geomeTRIC, fallback to local builder
# -------------------------
def _label_for_internal(ic: dict, atoms: list[str]) -> str:
    tp = ic['type']
    inds = ic['inds']
    if tp == 'bond':
        i, j = inds
        return f"bond:{i+1}-{j+1}:{atoms[i]}-{atoms[j]}"
    if tp == 'angle':
        i, j, k = inds
        return f"angle:{i+1}-{j+1}-{k+1}:{atoms[i]}-{atoms[j]}-{atoms[k]}"
    if tp == 'dihedral':
        i, j, k, l = inds
        return f"dihedral:{i+1}-{j+1}-{k+1}-{l+1}:{atoms[i]}-{atoms[j]}-{atoms[k]}-{atoms[l]}"
    return f"unknown:{inds}"


def _value_for_internal(ic: dict, coords: np.ndarray):
    if ic['type'] == 'bond':
        i, j = ic['inds']
        return float(np.linalg.norm(coords[i] - coords[j]))
    elif ic['type'] == 'angle':
        i, j, k = ic['inds']
        return _angle_deg(coords[i], coords[j], coords[k])
    elif ic['type'] == 'dihedral':
        i, j, k, l = ic['inds']
        return _dihedral_deg(coords[i], coords[j], coords[k], coords[l])
    else:
        return float('nan')


def _build_internals_geometric_if_available(atoms, coords):
    """
    Best-effort attempt to build redundant internals using geomeTRIC.
    The geomeTRIC API surface can vary; we try a few common helpers.
    If geomeTRIC is not available or the attempt fails, returns None.
    """
    if _GEOMETRIC is None:
        return None

    try:
        # Many geomeTRIC workflows expose functionality to produce Z-matrix
        # or redundant internals via classes. We attempt a generic approach:
        # - If the module provides a function or class to compute RICs, use it.
        # This is best-effort: if it fails we fallback to the local builder.
        mod = _GEOMETRIC

        # Common candidate call patterns we try (in order):
        candidates = [
            ("RedundantInternalCoordinates", True),
            ("Internals", True),
            ("internal_coordinates", False),
            ("build_redundant_internals", False),
            ("connectivity", False),
        ]

        for name, is_class in candidates:
            if hasattr(mod, name):
                member = getattr(mod, name)
                try:
                    if is_class:
                        # Try to instantiate with (atoms, coords) or (symbols, xyz)
                        try:
                            inst = member(atoms, coords)
                        except Exception:
                            inst = member(symbols=atoms, coords=coords)
                        # try to get a canonical list of internals
                        if hasattr(inst, "internals"):
                            # assume internals is a list of tuples (type, inds)
                            raw = inst.internals
                        elif hasattr(inst, "get_internals"):
                            raw = inst.get_internals()
                        else:
                            raw = None
                        if raw:
                            # Convert raw into our expected dict format if possible
                            out = []
                            for r in raw:
                                # try to coerce common shapes:
                                if isinstance(r, tuple) and len(r) >= 2:
                                    # first entry may be type string
                                    if isinstance(r[0], str):
                                        tp = r[0].lower()
                                        inds = r[1]
                                    else:
                                        # assume a tuple of indices -> bond/angle/dihedral by length
                                        inds = r
                                        if len(inds) == 2:
                                            tp = "bond"
                                        elif len(inds) == 3:
                                            tp = "angle"
                                        elif len(inds) == 4:
                                            tp = "dihedral"
                                        else:
                                            continue
                                    out.append({"type": tp, "inds": tuple(int(x) for x in inds)})
                            if out:
                                return out
                    else:
                        # functional API: try calling with (atoms, coords)
                        try:
                            raw = member(atoms, coords)
                        except Exception:
                            raw = member(symbols=atoms, coords=coords)
                        # attempt to convert raw as above
                        if raw:
                            out = []
                            for r in raw:
                                if isinstance(r, dict) and 'type' in r and 'inds' in r:
                                    out.append({'type': r['type'], 'inds': tuple(r['inds'])})
                                elif isinstance(r, tuple):
                                    # assume tuple of indices
                                    inds = r
                                    if len(inds) == 2:
                                        tp = "bond"
                                    elif len(inds) == 3:
                                        tp = "angle"
                                    elif len(inds) == 4:
                                        tp = "dihedral"
                                    else:
                                        continue
                                    out.append({"type": tp, "inds": tuple(int(x) for x in inds)})
                            if out:
                                return out
                except Exception:
                    # try next candidate
                    continue
    except Exception:
        # Any unexpected error should not break execution; fallback will be used.
        traceback.print_exc()
        return None

    # No usable API found / conversion failed
    return None


def build_redundant_internals(atoms, coords, bond_scale=1.2):
    """
    Try geomeTRIC first (best-effort). If unavailable or unusable,
    fall back to a local simple builder based on covalent radii.
    Returned list format: [{'type': 'bond'|'angle'|'dihedral', 'inds': (...), 'value': float}, ...]
    """
    # Attempt geomeTRIC
    geome_list = _build_internals_geometric_if_available(atoms, coords)
    if geome_list is not None:
        # compute numeric values and return
        for ic in geome_list:
            if ic['type'] == 'bond':
                i, j = ic['inds']
                ic['value'] = float(np.linalg.norm(coords[i] - coords[j]))
            elif ic['type'] == 'angle':
                i, j, k = ic['inds']
                ic['value'] = _angle_deg(coords[i], coords[j], coords[k])
            elif ic['type'] == 'dihedral':
                i, j, k, l = ic['inds']
                ic['value'] = _dihedral_deg(coords[i], coords[j], coords[k], coords[l])
        print("Using geomeTRIC-based redundant-internal coordinate builder (best-effort).")
        return geome_list

    # Fallback: original in-repo builder (robust for many molecules)
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
            ic['value'] = _angle_deg(coords[i], coords[j], coords[k])
        elif ic['type'] == 'dihedral':
            i, j, k, l = ic['inds']
            ic['value'] = _dihedral_deg(coords[i], coords[j], coords[k], coords[l])
    print("Using fallback internal redundant-internal coordinate builder.")
    return internals


# -------------------------
# Geometric transformations for single-internal updates (approximate)
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
    cur_angle = _angle_deg(p[i], p[j], p[k])
    target = new_angle_deg
    dtheta = math.radians(target - cur_angle)
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
# We cache by xyz string, method and spin. When CBS parameters are changed we clear cache.
@lru_cache(maxsize=2000)
def _compute_cbs_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str, spin: int):
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
    scf_small, scf_big = scf_vals[0], scf_vals[1]
    corr_small, corr_big = corr_vals[0], corr_vals[1]

    E_corr_cbs = corr_big + a_corr * (corr_big - corr_small)
    E_hf_cbs = scf_big + b_hf * (scf_big - scf_small)

    E_cbs = E_hf_cbs + E_corr_cbs
    return float(E_cbs)


# Small wrapper for caller to manage cache clearing
def compute_cbs_energy_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str, spin: int):
    # clear underlying lru cache for CBS cached computations in case parameters changed
    return _compute_cbs_from_xyz_cached(xyz_string, method, a_corr, b_hf, bs1, bs2, int(spin))


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
                      basis_pair=None, spin: int = 0):
    """
    Sequential SQM-style optimizer that updates current_coords if parabolic minimum improves CBS energy.
    Returns atoms, final_coords, history(list of dicts), converged(bool) and internals_trace (list of dicts)
    """
    # compute extrapolation coefficients (same algebra as L-BFGS module)
    a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
    denom = math.exp(beta * x2_hf) - math.exp(beta * x1_hf)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    b_hf = math.exp(beta * x1_hf) / denom

    # If a different basis pair provided, use it
    if basis_pair is None:
        basis_pair = (basis_sets[0], basis_sets[1])
    bs1, bs2 = basis_pair

    # Build baseline internals (kept fixed as labels for reproducibility/tracing)
    baseline_internals = build_redundant_internals(atoms, coords)
    if not baseline_internals:
        raise RuntimeError("No internal coordinates found; check input geometry.")

    # Keep a list of labels and a canonical order for tracing
    baseline_labels = [_label_for_internal(ic, atoms) for ic in baseline_internals]
    baseline_inds = [tuple(ic['inds']) for ic in baseline_internals]
    baseline_types = [ic['type'] for ic in baseline_internals]

    # Use a working list that will be rebuilt during optimization (local builder)
    internals = [dict(ic) for ic in baseline_internals]

    current_coords = coords.copy()
    displacement_factor = fac_mult
    history = []
    converged = False

    # internals_trace: list per cycle mapping label -> value (raw float strings)
    internals_trace = []

    # record initial values (cycle 0)
    init_vals = {}
    for lbl, tp, inds in zip(baseline_labels, baseline_types, baseline_inds):
        ic = {'type': tp, 'inds': inds}
        val = _value_for_internal(ic, current_coords)
        init_vals[lbl] = val
    internals_trace.append(init_vals)

    # Clear LRU cache for CBS cached computations in case parameters changed
    try:
        _compute_cbs_from_xyz_cached.cache_clear()
    except Exception:
        pass

    for cycle in range(1, maxcycle + 1):
        # compute current energy
        cur_xyz = xyz_to_pyscf_string(atoms, current_coords)
        try:
            current_energy = compute_cbs_energy_from_xyz_cached(cur_xyz, method, a_corr, b_hf, bs1, bs2, spin)
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
                    E = compute_cbs_energy_from_xyz_cached(xyzs, method, a_corr, b_hf, bs1, bs2, spin)
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
                    # rebuild internals values on updated geometry (use fallback builder to refresh 'internals')
                    internals = build_redundant_internals(atoms, current_coords)
                    updated = True
                    # recompute current_energy to use for subsequent comparisons
                    current_energy = e_min
                    print("    geometry updated (improvement)")
                else:
                    print("    no improvement")
            except Exception as e:
                print(f"    error in parabolic fit for IC {ic.get('inds')}: {e}")

        # record cycle energy (after processing all internals)
        history.append({'cycle': cycle, 'energy': float(current_energy)})

        # record internals values for this cycle using baseline internals for stable rows
        cyc_vals = {}
        for lbl, tp, inds in zip(baseline_labels, baseline_types, baseline_inds):
            ic = {'type': tp, 'inds': inds}
            val = _value_for_internal(ic, current_coords)
            cyc_vals[lbl] = val
        internals_trace.append(cyc_vals)

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
# run_optimization API
# -------------------------
def run_optimization(params: dict, outputs_dir: Path):
    """
    Standard entry from opt_cli.prepare_options_from_params
    params should include:
      - input_xyz (path to file)
      - method (optional)
      - X1, X2, Xhf1, X2hf, beta (optional)
      - maxcycle (optional)
      - fac (optional)
      - spin (optional)
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

    # spin (propagate to PySCF)
    spin = int(params.get("spin", DEFAULT_SPIN))

    atoms, coords0, comment = read_xyz(input_xyz)
    print(f"Loaded {len(atoms)} atoms from {input_xyz}")
    print("Building redundant internal coordinates (initial)...")
    try:
        internals = build_redundant_internals(atoms, coords0)
    except Exception as e:
        raise RuntimeError(f"Failed to build redundant internals for initial geometry: {e}")
    print(f"Found {len(internals)} internal coordinates (bonds/angles/dihedrals).")

    # run optimizer
    atoms_out, coords_out, history, converged, baseline_labels, internals_trace = optimize_from_xyz(
        atoms,
        coords0,
        method=method,
        maxcycle=maxcycle,
        fac_mult=fac,
        x1=x1, x2=x2, x1_hf=x1hf, x2_hf=x2hf, beta=beta,
        basis_pair=basis_pair,
        spin=spin
    )

    # write outputs via writer helpers
    base = Path(input_xyz).stem
    prefix = f"{base}_SQM"
    cycles_file = write_cycle_energies(outputs_dir, prefix, history)
    xyz_file = write_final_xyz(outputs_dir, prefix, atoms_out, coords_out, history[-1]['energy'] if history else 0.0)

    # Print internals trace table to terminal (rows: internals, cols: cycle0..N)
    try:
        print("\nRedundant-internal coordinates trace (rows: internal, columns: cycle 0..):")
        # Header
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
    }
    out = Path(args.out)
    result = run_optimization(params, out)
    print("Done. Outputs written to:", out)