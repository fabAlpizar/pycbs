#!/usr/bin/env python3
"""
SQM-based optimizer module (geomeTRIC-backed implementation)

This variant uses geomeTRIC's PrimitiveInternalCoordinates when available
and falls back to a geometry-derived redundant internal set (bonds/angles/dihedrals)
if geomeTRIC does not expose usable internals for the provided Molecule.

Enhancements in this version:
 - Best-improvement subiterations per cycle (evaluate all internals, apply only the single best improvement).
 - ENERGY_ACCEPT_TOL and MIN_CURVATURE thresholds (reject tiny/noisy moves).
 - Per-evaluation diagnostics: SCF convergence checks, numeric gradient & curvature prints,
   predicted vs actual internal values, candidate selection debug prints.
 - A `debug` flag can be set in params to increase terminal output.
"""

from pathlib import Path
import math
import sys
from functools import lru_cache
import traceback

import numpy as np
from pycbs.writer import write_cycle_energies, write_final_xyz

# Require geomeTRIC's geometric module (we handle if import fails later)
try:
    from geometric.internal import PrimitiveInternalCoordinates
except Exception as e:
    raise ImportError("This SQM implementation requires the 'geometric' package "
                      "and specifically PrimitiveInternalCoordinates from "
                      "geometric.internal. Install geometric/geomeTRIC and retry.") from e

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
# Geometry-derived internals fallback
# -------------------------
def _covalent_radius(elem: str) -> float:
    return COVALENT_RADII.get(elem, DEFAULT_COV_RAD)


def generate_internals_from_geometry(atoms: list[str], coords: np.ndarray, scale: float = 1.2):
    """
    Build a simple redundant internal set from geometry:
      - bonds: distance < scale * (rcov_i + rcov_j)
      - angles: any triplet i - j - k where i and k are bonded to j (i < k to avoid duplicates)
      - dihedrals: any quadruplet i - j - k - l where i bonded to j, j bonded k, k bonded l
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
                # i - j - k, now extend k to l
                for l in bond_neighbors.get(k, []):
                    if l == j:
                        continue
                    inds = (i, j, k, l)
                    # canonical ordering to avoid duplicates
                    if inds not in diheds:
                        diheds.add(inds)
                        val = _dihedral_deg(coords[i], coords[j], coords[k], coords[l])
                        out.append({"type": "dihedral", "inds": inds, "value": float(val)})

    if not out:
        raise RuntimeError("Fallback geometry-based internals generation produced no internals (geometry suspicious).")
    return out


# -------------------------
# geomeTRIC PrimitiveInternalCoordinates integration (robust)
# -------------------------
def _is_sequence_of_ints(x):
    try:
        if isinstance(x, (list, tuple)):
            return all(isinstance(i, (int, np.integer)) for i in x)
        if isinstance(x, np.ndarray) and x.ndim == 1:
            return np.issubdtype(x.dtype, np.integer)
        return False
    except Exception:
        return False


def _extract_indices_from_obj(obj):
    # Try many plausible attribute names that may contain atom indices
    for attr in ('indices', 'atoms', 'atoms_idx', 'atom_indices', 'atom_ids', 'atom_idxs', 'idx', 'i', 'a'):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple, np.ndarray)):
                return tuple(int(x) for x in np.asarray(val).reshape(-1))
    # If object is a simple container (like numpy array)
    if isinstance(obj, (list, tuple, np.ndarray)):
        try:
            return tuple(int(x) for x in list(obj))
        except Exception:
            pass
    return None


def build_redundant_internals_geometric(atoms: list[str], coords: np.ndarray):
    """
    Build RICs strictly using geomeTRIC's PrimitiveInternalCoordinates if possible,
    otherwise fallback to geometry-derived internals.
    Returns list of dicts: {'type': 'bond'|'angle'|'dihedral', 'inds': tuple(indices), 'value': float}
    where indices are zero-based integers.
    """
    pic = None
    tried = []

    # Preferred: construct a geomeTRIC Molecule object if available.
    try:
        from geometric.molecule import Molecule as GeometricMolecule

        gm = GeometricMolecule()
        gm.Data = {
            "resname": ["UNK"] * len(atoms),
            "resid": [0] * len(atoms),
            "elem": list(atoms),
            "bonds": [],
            "name": "pycbs_tmp",
            # geomeTRIC expects a list of frames; supply one frame (Angstroms)
            "xyzs": [coords.tolist()]
        }
        pic = PrimitiveInternalCoordinates(gm)
    except Exception as exc_gm:
        tried.append((("GeometricMolecule",), exc_gm))
        pic = None

    # If we didn't get a pic yet, try several plausible constructor signatures
    if pic is None:
        for args in (
            (atoms, coords.tolist()),
            (atoms, coords),
            (coords.tolist(), atoms),
            (coords, atoms),
        ):
            try:
                pic = PrimitiveInternalCoordinates(*args)
                break
            except Exception as exc:
                tried.append((args, exc))
                pic = None

    if pic is None:
        msg = "Failed to instantiate PrimitiveInternalCoordinates with tried signatures.\n"
        for args, exc in tried:
            msg += f" Tried args={args!r} -> {type(exc).__name__}: {exc}\n"
        # Rather than aborting outright, try a geometry fallback
        # print diagnostic and proceed to fallback
        print(msg)
        print("Proceeding with geometry-derived internals fallback.")
        return generate_internals_from_geometry(atoms, coords)

    # Now attempt to extract the internal list.
    raw_ints = None

    # Try well-known attribute/method names first
    for name in ("intcos", "internals", "primitive_internals", "intcos_list", "internallist", "prims", "primitives"):
        if hasattr(pic, name):
            raw_ints = getattr(pic, name)
            if callable(raw_ints):
                try:
                    raw_ints = raw_ints()
                except Exception:
                    pass
            break

    if raw_ints is None:
        for meth in ("get_internals", "get_intcos", "get_primitive_internals", "as_intcos", "to_intcos"):
            if hasattr(pic, meth) and callable(getattr(pic, meth)):
                try:
                    raw_ints = getattr(pic, meth)()
                except Exception:
                    raw_ints = None
                if raw_ints is not None:
                    break

    # If still not found: inspect public attributes and pick the first attribute that looks like internals
    if raw_ints is None:
        candidates = []
        for attr in sorted(set(dir(pic))):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(pic, attr)
            except Exception:
                continue
            # list/tuple candidate
            if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
                first = val[0]
                # case: sequence of integer-index sequences or 2D int ndarray
                if isinstance(first, (list, tuple)) and all(isinstance(ii, (int, np.integer)) for ii in first):
                    candidates.append((attr, val))
                    continue
                if isinstance(val, np.ndarray) and val.ndim == 2 and np.issubdtype(val.dtype, np.integer):
                    candidates.append((attr, val))
                    continue
                # case: list of objects that probably wrap indices (have .indices/.atoms)
                if hasattr(first, "indices") or hasattr(first, "atoms") or hasattr(first, "atoms_idx") or hasattr(first, "type") or hasattr(first, "kind"):
                    candidates.append((attr, val))
                    continue
            # also consider single objects that are array-like
            if hasattr(val, "__len__") and not isinstance(val, (str, bytes)) and len(val) > 0:
                try:
                    first = val[0]
                    if isinstance(first, (list, tuple)) and all(isinstance(ii, (int, np.integer)) for ii in first):
                        candidates.append((attr, val))
                except Exception:
                    pass
        if candidates:
            # prefer those with ndarray 2D integer arrays or direct sequence of int-tuples
            chosen_attr, chosen_val = None, None
            for attr, val in candidates:
                if isinstance(val, np.ndarray) and val.ndim == 2 and np.issubdtype(val.dtype, np.integer):
                    chosen_attr, chosen_val = attr, val
                    break
            if chosen_attr is None:
                chosen_attr, chosen_val = candidates[0]
            raw_ints = chosen_val
            # Diagnostic print so you know which attribute was chosen
            print(f"Warning: PrimitiveInternalCoordinates did not expose a standard attribute; "
                  f"using '{chosen_attr}' (type={type(chosen_val).__name__}) as internals candidate.")

    if raw_ints is None:
        # No candidate found: fallback to geometry
        print("PrimitiveInternalCoordinates did not expose any suitable internals attribute.")
        print("Proceeding with geometry-derived internals fallback.")
        return generate_internals_from_geometry(atoms, coords)

    # Normalize raw_ints entries into a canonical format
    out = []
    for r in raw_ints:
        # Possibilities:
        # - r is a tuple/list of ints -> infer type by length
        # - r is (type_str, indices)
        # - r is an object with attributes .type and .atoms or .indices
        try:
            # numpy 2D array row -> convert
            if isinstance(r, np.ndarray) and r.ndim == 1 and np.issubdtype(r.dtype, np.integer):
                inds = tuple(int(x) for x in r.tolist())
                if len(inds) == 2:
                    tp = "bond"
                elif len(inds) == 3:
                    tp = "angle"
                elif len(inds) == 4:
                    tp = "dihedral"
                else:
                    continue
                val = _value_for_internal(tp, inds, coords)
                out.append({"type": tp, "inds": inds, "value": val})
                continue
        except Exception:
            pass

        try:
            if isinstance(r, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in r):
                inds = tuple(int(x) for x in r)
                if len(inds) == 2:
                    tp = "bond"
                elif len(inds) == 3:
                    tp = "angle"
                elif len(inds) == 4:
                    tp = "dihedral"
                else:
                    continue
                val = _value_for_internal(tp, inds, coords)
                out.append({"type": tp, "inds": inds, "value": val})
                continue
        except Exception:
            pass

        # If r is a pair like ('bond', (i,j))
        try:
            if isinstance(r, (list, tuple)) and len(r) >= 2 and isinstance(r[0], str):
                tp = str(r[0]).lower()
                inds_cand = r[1]
                if isinstance(inds_cand, (list, tuple, np.ndarray)):
                    inds = tuple(int(x) for x in np.asarray(inds_cand).reshape(-1))
                    val = _value_for_internal(tp, inds, coords)
                    out.append({"type": tp, "inds": inds, "value": val})
                    continue
        except Exception:
            pass

        # If r is an object with attributes
        try:
            tp_attr = getattr(r, "type", None) or getattr(r, "kind", None) or getattr(r, "label", None)
            idx = _extract_indices_from_obj(r)
            if tp_attr and idx is not None:
                tp = str(tp_attr).lower()
                inds = tuple(int(x) for x in idx)
                val = _value_for_internal(tp, inds, coords)
                out.append({"type": tp, "inds": inds, "value": val})
                continue
        except Exception:
            pass

        # If r is an object representing indices under some attribute name
        try:
            idx = _extract_indices_from_obj(r)
            if idx is not None:
                inds = tuple(int(x) for x in idx)
                if len(inds) == 2:
                    tp = "bond"
                elif len(inds) == 3:
                    tp = "angle"
                elif len(inds) == 4:
                    tp = "dihedral"
                else:
                    continue
                val = _value_for_internal(tp, inds, coords)
                out.append({"type": tp, "inds": inds, "value": val})
                continue
        except Exception:
            pass

        # If none matched, skip (but keep strict behavior)
        raise RuntimeError(f"Unrecognized internal coordinate entry from geometric: {r!r}")

    if not out:
        # geomeTRIC returned an empty list or we couldn't normalize: fallback
        print("geometric returned an empty list of internals; using geometry-derived fallback.")
        return generate_internals_from_geometry(atoms, coords)

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
@lru_cache(maxsize=2000)
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

        # Check SCF convergence; if not converged, raise so caller treats evaluation as failed
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

    scf_small, scf_big = scf_vals[0], scf_vals[1]
    corr_small, corr_big = corr_vals[0], corr_vals[1]

    E_corr_cbs = corr_big + a_corr * (corr_big - corr_small)
    E_hf_cbs = scf_big + b_hf * (scf_big - scf_small)
    E_cbs = E_hf_cbs + E_corr_cbs
    return float(E_cbs)


def compute_cbs_energy_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str, spin: int):
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
                      basis_pair=None, spin: int = 0, debug: bool = False):
    """
    Strictly geomeTRIC-backed SQM-style optimizer with best-improvement subiterations.
    Returns atoms, final_coords, history(list), converged(bool), baseline_labels, internals_trace

    debug: when True, prints detailed diagnostics for each internal evaluation.
    """
    # acceptance / diagnostics thresholds (local)
    ENERGY_ACCEPT_TOL = 1e-6  # Ha (per-move acceptance)
    MIN_CURVATURE = 1e-10     # minimal convex curvature required
    # numeric gradient print thresholds (for developer visibility)
    GRAD_PRINT_THRESHOLD = 1e-9

    a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
    denom = math.exp(beta * x2_hf) - math.exp(beta * x1_hf)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    b_hf = math.exp(beta * x1_hf) / denom

    if basis_pair is None:
        basis_pair = (basis_sets[0], basis_sets[1])
    bs1, bs2 = basis_pair

    # Obtain canonical baseline internals using geomeTRIC (or fallback)
    baseline = build_redundant_internals_geometric(atoms, coords)
    if not baseline:
        raise RuntimeError("No internals extracted by geometric or fallback; aborting.")

    baseline_labels = [_label_internal(ic['type'], ic['inds'], atoms) for ic in baseline]
    baseline_kinds = [ic['type'] for ic in baseline]
    baseline_inds = [ic['inds'] for ic in baseline]

    # Working internals (will be refreshed after geometry updates by re-calling geometric/fallback)
    internals = [dict(ic) for ic in baseline]

    current_coords = coords.copy()
    displacement_factor = fac_mult
    history = []
    converged = False

    internals_trace = []
    # cycle 0 initial values
    init_map = {
        lbl: _value_for_internal(tp, inds, current_coords)
        for lbl, tp, inds in zip(baseline_labels, baseline_kinds, baseline_inds)
    }
    internals_trace.append(init_map)

    try:
        _compute_cbs_from_xyz_cached.cache_clear()
    except Exception:
        pass

    for cycle in range(1, maxcycle + 1):
        cur_xyz = xyz_to_pyscf_string(atoms, current_coords)
        try:
            current_energy = compute_cbs_energy_from_xyz_cached(cur_xyz, method, a_corr, b_hf, bs1, bs2, spin)
        except Exception as e:
            raise RuntimeError(f"CBS evaluation at cycle start failed: {e}")

        if debug:
            print(f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}, current E = {current_energy:.10f} Ha")
        else:
            print(f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}, current E = {current_energy:.10f} Ha")

        # -----------------------
        # Best-improvement subiteration:
        # evaluate ALL internals (but DO NOT apply), pick and APPLY the single best improvement,
        # repeat until no acceptable candidate remains.
        # -----------------------
        applied_any = False
        while True:
            best_candidate = None  # tuple (deltaE, e_min, best_coords, ic, x_min_disp, curvature, grad0)
            # Evaluate every internal but do NOT apply them yet
            for ic in internals:
                tp = ic['type']; inds = ic['inds']
                # build displacements ds
                if tp == 'bond':
                    base = ic.get('value', _value_for_internal(tp, inds, current_coords))
                    ds = np.array([-2*displacement_factor*base, -1*displacement_factor*base, 0.0,
                                   1*displacement_factor*base, 2*displacement_factor*base], dtype=float)
                elif tp in ('angle', 'dihedral'):
                    base = ic.get('value', _value_for_internal(tp, inds, current_coords))
                    ddeg = 2.0 * (displacement_factor * 100.0)
                    ds = np.array([-2*ddeg, -1*ddeg, 0.0, 1*ddeg, 2*ddeg], dtype=float)
                else:
                    ds = np.array([0.0])

                es = []
                coords_list = []
                scf_failed_any = False
                for d in ds:
                    try:
                        if tp == 'bond':
                            i,j = inds
                            new_val = ic.get('value', _value_for_internal(tp, inds, current_coords)) + d
                            new_coords = apply_bond_change(current_coords, i, j, new_val, atoms)
                        elif tp == 'angle':
                            i,j,k = inds
                            new_val = ic.get('value', _value_for_internal(tp, inds, current_coords)) + d
                            new_coords = apply_angle_change(current_coords, i, j, k, new_val)
                        elif tp == 'dihedral':
                            i,j,k,l = inds
                            new_val = ic.get('value', _value_for_internal(tp, inds, current_coords)) + d
                            new_coords = apply_dihedral_change(current_coords, i, j, k, l, new_val)
                        else:
                            new_coords = current_coords.copy()
                        xyzs = xyz_to_pyscf_string(atoms, new_coords)
                        E = compute_cbs_energy_from_xyz_cached(xyzs, method, a_corr, b_hf, bs1, bs2, spin)
                        es.append(float(E)); coords_list.append(new_coords)
                    except Exception as exc:
                        # mark failed evaluations with inf and continue; we will skip if all fail
                        es.append(float('inf')); coords_list.append(None)
                        scf_failed_any = True
                        if debug:
                            print(f"    eval failed for IC {tp} {inds} displacement {d}: {exc}")

                es = np.array(es, dtype=float)
                if np.all(np.isinf(es)):
                    if debug:
                        print(f"  All evaluations failed for IC {tp} {inds}; skipping")
                    continue

                # parabolic fit
                x_min_disp, e_min = parabolic_minimum(ds, es)
                idx_best = int(np.nanargmin(es))
                best_coords = coords_list[idx_best]

                # numeric curvature & gradient check (centered finite differences)
                curvature = None
                grad0 = None
                try:
                    # find index of zero displacement (exact match)
                    idx0 = int(np.where(np.isclose(ds, 0.0))[0][0])
                    # ensure we have neighbors for central diff
                    if idx0 - 1 >= 0 and idx0 + 1 < len(es):
                        Eminus = es[idx0-1]; E0 = es[idx0]; Eplus = es[idx0+1]
                        h = ds[idx0+1] - ds[idx0]
                        grad0 = (Eplus - Eminus) / (2.0 * h)
                        curvature = (Eplus + Eminus - 2.0 * E0) / (h * h)
                except Exception:
                    grad0 = None
                    curvature = None

                # candidate delta relative to current_energy
                deltaE = current_energy - e_min  # positive => improvement

                # debug prints for this internal's sampling and numeric checks
                if debug:
                    cur_val = ic.get('value', _value_for_internal(tp, inds, current_coords))
                    pred_val = cur_val + x_min_disp
                    sampled_best_val = None
                    try:
                        sampled_best_val = _value_for_internal(tp, inds, coords_list[idx_best]) if coords_list[idx_best] is not None else None
                    except Exception:
                        sampled_best_val = None
                    print(f"  IC {tp} {inds}: cur={cur_val:.6f}, ds=[{', '.join(f'{x:.4g}' for x in ds)}], "
                          f"E_samples=[{', '.join(f'{e:.6f}' for e in es)}], x_min_disp={x_min_disp:.6f}, e_min={e_min:.10f}")
                    if grad0 is not None:
                        print(f"    numeric grad @0 = {grad0:.3e} Ha/unit, curvature = {curvature:.3e} Ha/unit^2")
                    else:
                        print("    numeric grad/curvature not available (insufficient sampling)")

                    if sampled_best_val is not None:
                        print(f"    predicted new internal value (from x_min_disp) = {pred_val:.6f}, best sampled internal value = {sampled_best_val:.6f}")
                    if scf_failed_any:
                        print("    NOTE: some SCF evaluations failed for this IC sampling (see above)")

                # selection criterion: convex parabolic fit, sufficient improvement, and best delta
                if best_coords is not None and curvature is not None and curvature > MIN_CURVATURE and deltaE > ENERGY_ACCEPT_TOL:
                    if best_candidate is None or deltaE > best_candidate[0]:
                        best_candidate = (deltaE, e_min, best_coords, ic, x_min_disp, curvature, grad0)

            # end for internals: apply best candidate if exists
            if best_candidate is None:
                if debug:
                    print("  No acceptable candidate found in this subiteration.")
                break  # nothing worthwhile to apply
            # apply the best candidate
            deltaE, e_min, best_coords, chosen_ic, x_min_disp, curvature, grad0 = best_candidate
            # final diagnostic before apply
            if debug:
                print(f"  Applying best IC {chosen_ic['type']} {chosen_ic['inds']}: ΔE = {deltaE:.3e} Ha, curvature={curvature:.3e}, grad0={grad0}")
            else:
                print(f"  Applied best IC {chosen_ic['type']} {chosen_ic['inds']}: ΔE = {deltaE:.3e} Ha (curvature={curvature:.3e})")
            current_coords = best_coords.copy()
            # refresh working internals for updated geometry
            internals = build_redundant_internals_geometric(atoms, current_coords)
            # update current energy
            current_energy = e_min
            applied_any = True
            # loop: continue searching for another best internal on the updated geometry

        # End subiteration while

        # record cycle energy and internals trace
        history.append({'cycle': cycle, 'energy': float(current_energy)})

        cyc_map = {
            lbl: _value_for_internal(tp, inds, current_coords)
            for lbl, tp, inds in zip(baseline_labels, baseline_kinds, baseline_inds)
        }
        internals_trace.append(cyc_map)

        # convergence check: cycle-level energy difference
        if cycle > 1:
            ediff = abs(history[-1]['energy'] - history[-2]['energy'])
            print(f"  ΔE since last cycle: {ediff:.4e} Ha")
            if ediff < ENERGY_CRIT:
                # additionally, if debug show gradient-like diagnostics summary from last subiteration (best_candidate info missing),
                print("Converged by energy criterion")
                converged = True
                break

        displacement_factor *= CUT

    return atoms, current_coords, history, converged, baseline_labels, internals_trace


# -------------------------
# run_optimization API
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

    atoms, coords0, comment = read_xyz(input_xyz)
    print(f"Loaded {len(atoms)} atoms from {input_xyz}")
    print("Extracting redundant internal coordinates (geometric PrimitiveInternalCoordinates or fallback)...")
    internals = build_redundant_internals_geometric(atoms, coords0)
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
        debug=debug
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
    }
    out = Path(args.out)
    result = run_optimization(params, out)
    print("Done. Outputs written to:", out)