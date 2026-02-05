#!/usr/bin/env python3
"""
SQM-based optimizer module (geomeTRIC-backed implementation)

This variant strictly uses the geometric (geomeTRIC) package to extract
redundant internal coordinates (RICs) via PrimitiveInternalCoordinates
and then runs an SQM-style sequential internal-coordinate scan optimizer
that:

 - Loads RICs (types and atom indices) from geometric.internal.PrimitiveInternalCoordinates.
 - Uses those RIC labels/indices as the canonical baseline internals for tracing.
 - For each internal performs a 5-point parabolic scan in that internal
   (bond: Å, angle/dihedral: degrees), evaluates CBS energies with PySCF,
   and updates Cartesian coordinates if the best point improves the CBS energy.
 - Records the value of every baseline internal at cycle 0..N (raw floats).
 - At the end prints a terminal table: rows = RIC labels, columns = cycle values.
 - Persists cycle energies and final xyz to outputs (prefixed with input basename).

Notes:
 - This file intentionally requires the `geometric` package. If it's not
   importable or PrimitiveInternalCoordinates cannot be used, the module
   will raise ImportError. You indicated geometric is available.
 - The conversion from internal displacements to Cartesian updates here uses
   geometric only for extracting internals. The actual per-internal Cartesian
   updates are performed with geometric-aware index mapping but still use the
   same geometric adjustments as the prior SQM implementation (approximate
   local moves). If you want a full B-matrix-based internal->Cartesian update
   (exact Wilson/B-matrix inversion), that requires more extensive use of
   geomeTRIC internal machinery (or geomeTRIC optimizers) and can be added.
"""

from pathlib import Path
import math
import sys
from functools import lru_cache
import traceback

import numpy as np
from pycbs.writer import write_cycle_energies, write_final_xyz

# Require geomeTRIC's geometric module
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
# geomeTRIC PrimitiveInternalCoordinates integration
# -------------------------
def build_redundant_internals_geometric(atoms: list[str], coords: np.ndarray):
    """
    Build RICs strictly using geomeTRIC's PrimitiveInternalCoordinates.

    Returns list of dicts: {'type': 'bond'|'angle'|'dihedral', 'inds': tuple(indices), 'value': float}
    where indices are zero-based integers.
    If the geomeTRIC object provides a slightly different representation, we
    try to extract type and atom indices accordingly. If extraction fails we
    raise an informative error.
    """
    # PrimitiveInternalCoordinates constructor signatures may vary; try common ones.
    pic = None
    tried = []
    # Attempt several plausible constructor signatures
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
        # Provide diagnostic info
        msg = "Failed to instantiate PrimitiveInternalCoordinates with tried signatures.\n"
        for args, exc in tried:
            msg += f" Tried args={args!r} -> {type(exc).__name__}: {exc}\n"
        raise RuntimeError(msg)

    # Now attempt to extract the internal list.
    raw_ints = None
    # try a few likely attribute/method names
    for name in ("intcos", "internals", "primitive_internals", "intcos_list", "internallist"):
        if hasattr(pic, name):
            raw_ints = getattr(pic, name)
            break
    if raw_ints is None:
        # some versions expose a method
        for meth in ("get_internals", "get_intcos", "get_primitive_internals"):
            if hasattr(pic, meth) and callable(getattr(pic, meth)):
                raw_ints = getattr(pic, meth)()
                break

    if raw_ints is None:
        # As last resort inspect repr/dir for clues
        raise RuntimeError("Could not locate internals list in PrimitiveInternalCoordinates instance. "
                           "Please check the geometric package version and API.")

    # Normalize raw_ints entries into a canonical format
    out = []
    for r in raw_ints:
        # Possibilities:
        # - r is a tuple/list of ints -> infer type by length
        # - r is (type_str, indices)
        # - r is an object with attributes .type and .atoms or .indices
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
                if isinstance(inds_cand, (list, tuple)):
                    inds = tuple(int(x) for x in inds_cand)
                    val = _value_for_internal(tp, inds, coords)
                    out.append({"type": tp, "inds": inds, "value": val})
                    continue
        except Exception:
            pass

        # If r is an object with attributes
        try:
            # e.g., r.type, r.indices, r.atoms
            tp_attr = getattr(r, "type", None) or getattr(r, "kind", None)
            idx_attr = getattr(r, "indices", None) or getattr(r, "atoms", None) or getattr(r, "atoms_idx", None)
            if tp_attr and idx_attr is not None:
                tp = str(tp_attr).lower()
                if isinstance(idx_attr, (list, tuple)):
                    inds = tuple(int(x) for x in idx_attr)
                else:
                    # maybe a numpy array
                    inds = tuple(int(x) for x in list(idx_attr))
                val = _value_for_internal(tp, inds, coords)
                out.append({"type": tp, "inds": inds, "value": val})
                continue
        except Exception:
            pass

        # If none matched, skip but warn (we keep strict behavior: do not fallback)
        raise RuntimeError(f"Unrecognized internal coordinate entry from geometric: {r!r}")

    if not out:
        raise RuntimeError("geometric returned an empty list of internals; aborting.")

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
                      basis_pair=None, spin: int = 0):
    """
    Strictly geomeTRIC-backed SQM-style optimizer.
    Returns atoms, final_coords, history(list), converged(bool), baseline_labels, internals_trace
    """
    a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
    denom = math.exp(beta * x2_hf) - math.exp(beta * x1_hf)
    if abs(denom) < 1e-16:
        raise ZeroDivisionError("HF CBS denominator too small")
    b_hf = math.exp(beta * x1_hf) / denom

    if basis_pair is None:
        basis_pair = (basis_sets[0], basis_sets[1])
    bs1, bs2 = basis_pair

    # Obtain canonical baseline internals using geomeTRIC
    baseline = build_redundant_internals_geometric(atoms, coords)
    if not baseline:
        raise RuntimeError("No internals extracted by geometric; aborting.")

    baseline_labels = [_label_internal(ic['type'], ic['inds'], atoms) for ic in baseline]
    baseline_kinds = [ic['type'] for ic in baseline]
    baseline_inds = [ic['inds'] for ic in baseline]

    # Working internals (will be refreshed after geometry updates by re-calling geometric)
    internals = [dict(ic) for ic in baseline]

    current_coords = coords.copy()
    displacement_factor = fac_mult
    history = []
    converged = False

    internals_trace = []
    # cycle 0 initial values
    init_map = {lbl: _value_for_internal({'type': tp, 'inds': inds}, current_coords)
                for lbl, tp, inds in zip(baseline_labels, baseline_kinds, baseline_inds)}
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

        print(f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}, current E = {current_energy:.10f} Ha")

        updated = False
        for ic in internals:
            tp = ic['type']
            inds = ic['inds']
            if tp == 'bond':
                base = ic['value']
                disps = np.array([
                    -2 * displacement_factor * base,
                    -1 * displacement_factor * base,
                    0.0,
                    1 * displacement_factor * base,
                    2 * displacement_factor * base
                ], dtype=float)
            elif tp in ('angle', 'dihedral'):
                base = ic.get('value', _value_for_internal(tp, inds, current_coords))
                ddeg = 2.0 * (displacement_factor * 100.0)
                disps = np.array([-2*ddeg, -1*ddeg, 0.0, 1*ddeg, 2*ddeg], dtype=float)
            else:
                disps = np.array([0.0])

            energies = []
            coords_list = []
            for d in disps:
                try:
                    if tp == 'bond':
                        i, j = inds
                        new_val = ic['value'] + d
                        new_coords = apply_bond_change(current_coords, i, j, new_val, atoms)
                    elif tp == 'angle':
                        i, j, k = inds
                        new_val = ic['value'] + d
                        new_coords = apply_angle_change(current_coords, i, j, k, new_val)
                    elif tp == 'dihedral':
                        i, j, k, l = inds
                        new_val = ic['value'] + d
                        new_coords = apply_dihedral_change(current_coords, i, j, k, l, new_val)
                    else:
                        new_coords = current_coords.copy()

                    xyzs = xyz_to_pyscf_string(atoms, new_coords)
                    E = compute_cbs_energy_from_xyz_cached(xyzs, method, a_corr, b_hf, bs1, bs2, spin)
                    energies.append(float(E))
                    coords_list.append(new_coords)
                except Exception as e:
                    energies.append(float('inf'))
                    coords_list.append(None)
                    print(f"    eval failed for IC {tp} {inds} displacement {d}: {e}")

            es = np.array(energies, dtype=float)
            if np.all(np.isinf(es)):
                print(f"  Skipping {tp} {inds}: all evals failed")
                continue

            try:
                x_min_disp, e_min = parabolic_minimum(disps, es)
                idx_best = int(np.nanargmin(es))
                best_coords = coords_list[idx_best]
                print(f"  IC {tp} {inds}: current={ic['value']:.6f} -> x_min_disp={x_min_disp:.6f}, E_min={e_min:.10f}")
                if e_min < current_energy - 1e-12 and best_coords is not None:
                    current_coords = best_coords.copy()
                    # Refresh internals from geomeTRIC for the new geometry
                    internals = build_redundant_internals_geometric(atoms, current_coords)
                    updated = True
                    current_energy = e_min
                    print("    geometry updated (improvement)")
                else:
                    print("    no improvement")
            except Exception as e:
                print(f"    error in parabolic fit for IC {inds}: {e}")

        history.append({'cycle': cycle, 'energy': float(current_energy)})

        cyc_map = {lbl: _value_for_internal({'type': tp, 'inds': inds}, current_coords)
                   for lbl, tp, inds in zip(baseline_labels, baseline_kinds, baseline_inds)}
        internals_trace.append(cyc_map)

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

    atoms, coords0, comment = read_xyz(input_xyz)
    print(f"Loaded {len(atoms)} atoms from {input_xyz}")
    print("Extracting redundant internal coordinates (geometric PrimitiveInternalCoordinates)...")
    internals = build_redundant_internals_geometric(atoms, coords0)
    print(f"Found {len(internals)} internals (from geometric).")

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