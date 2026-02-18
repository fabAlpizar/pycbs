#!/usr/bin/env python3
"""
SQM-based optimizer module (ALL-PER-CYCLE approach)

OPTIMIZATION STRATEGY: ALL-PER-CYCLE APPROACH
==============================================

This optimizer performs SEQUENTIAL QUADRATIC MINIMIZATION (SQM) with an
ALL-PER-CYCLE update strategy:

STEP-BY-STEP WORKFLOW:
======================

1. INITIAL SETUP & READING
   - Reads XYZ file (atomic symbols + Cartesian coordinates)
   - Generates REDUNDANT INTERNAL COORDINATES (RICs):
     * Bonds: based on covalent radii cutoff
     * Angles: i-j-k where (i,j) and (j,k) are bonded
     * Dihedrals: i-j-k-l where (i,j), (j,k), (k,l) are bonded

2. CBS ENERGY CALCULATION (Composite Basis Set approach)
   - Evaluates energy at TWO basis sets: cc-pVDZ (small) and cc-pVTZ (large)
   - For each basis:
     * Performs RHF-SCF calculation (Hartree-Fock)
     * Calculates correlation energy (CCSD(T) or MP2)
   - Extrapolates to CBS limit using:
     * Exponential formula for correlation: E_corr(X) = E_∞ + a*exp(-β*X)
     * Exponential formula for HF: E_HF(X) = E_∞ + b*exp(-β*X)
     * Final CBS energy: E_CBS = E_HF_CBS + E_corr_CBS

3. PER-CYCLE OPTIMIZATION LOOP (ALL-PER-CYCLE STRATEGY)
   ======================================================

   For each optimization cycle:

   a) EVALUATE ALL INTERNAL COORDINATES
      - For EACH internal coordinate (bond, angle, dihedral):
        * Sample 3 points: -displacement, 0, +displacement
        * Calculate CBS energy at each point
        * Fit parabola through 3 points
        * Extract predicted minimum (x_min, E_min) and curvature

   b) UPDATE ALL INTERNALS THAT IMPROVE ENERGY
      - For each internal coordinate:
        * Check if predicted energy drop exceeds threshold (ENERGY_ACCEPT_TOL)
        * If YES: Calculate new coordinates using predicted minimum
        * Validate by computing actual CBS energy at that geometry
        * If validation confirms improvement: APPLY THE CHANGE
        * If validation fails: SKIP this internal
      - Internal coordinates are REGENERATED after EACH change (accounts for coupling)
      - Process continues until all profitable internals are attempted

   c) CONVERGENCE CHECK
      - Monitor total energy change in this cycle
      - Reduce displacement factor (multiply by 0.75) for next cycle
      - Check if ΔE < 1e-8 Ha → converged
      - Check if max cycles reached → stop

4. GEOMETRIC UPDATES (Local approximate moves)
   - Bond changes: Mass-weighted displacement
     * p_i -> p_i - w_i * Δr * direction
     * p_j -> p_j + (1-w_i) * Δr * direction
     * weights based on atomic masses

   - Angle changes: Rodrigues rotation
     * Rotate atom i around central atom j
     * Uses rotation axis and angle from geometry

   - Dihedral changes: Rodrigues rotation
     * Rotate atom i around j-k bond axis
     * Maintains bond lengths

5. OUTPUT GENERATION
   - Saves per-cycle XYZ geometries (cycle_000_initial, cycle_001, ...)
   - Saves final optimized XYZ
   - Saves energy vs cycle plot (with both PySCF and CBS energies)
   - Prints internal coordinate trace (values per cycle)

KEY DIFFERENCES FROM SINGLE-BEST STRATEGY:
============================================
Single-Best: Evaluates all → picks ONE → applies → re-evaluates
All-Per-Cycle: Evaluates all → APPLIES EACH profitable one → regenerates → repeats

Advantages of ALL-PER-CYCLE:
  ✓ Faster convergence (multiple improvements per cycle)
  ✓ More efficient for weakly-coupled coordinates
  ✓ Better for large molecules

Disadvantages of ALL-PER-CYCLE:
  ✗ Risk of unphysical geometries (coordinate coupling)
  ✗ Coordination effects between changes not fully accounted for
  ✗ May require more careful validation

Usage:
    python sqm_optimizer_all_per_cycle.py -i input.xyz --out PyCBS-OUTPUTS --workers 4 --debug
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

MAXCYCLE_DEFAULT = 70
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


def write_xyz(filename, atoms, coords, comment=""):
    """Write geometry to XYZ file"""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for a, c in zip(atoms, coords):
            f.write(f"{a} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n")


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
        return f"bond:{i + 1}-{j + 1}:{atoms[i]}-{atoms[j]}"
    if kind == 'angle':
        i, j, k = inds
        return f"angle:{i + 1}-{j + 1}-{k + 1}:{atoms[i]}-{atoms[j]}-{atoms[k]}"
    if kind == 'dihedral':
        i, j, k, l = inds
        return f"dihedral:{i + 1}-{j + 1}-{k + 1}-{l + 1}:{atoms[i]}-{atoms[j]}-{atoms[k]}-{atoms[l]}"
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
@lru_cache(maxsize=10000)
def _compute_cbs_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str,
                                 spin: int):
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


def compute_cbs_energy_from_xyz_cached(xyz_string: str, method: str, a_corr: float, b_hf: float, bs1: str, bs2: str,
                                       spin: int):
    return _compute_cbs_from_xyz_cached(xyz_string, method, a_corr, b_hf, bs1, bs2, int(spin))


# ---- HELPER FUNCTION: Compute PySCF HF energies for both bases ----
def compute_pyscf_hf_energies(atoms, coords, bs1, bs2, spin, pyscf_threads):
    """Compute RHF-SCF energies for both basis sets (for CSV logging)"""
    pyscf_energies = {}
    try:
        xyz_str = xyz_to_pyscf_string(atoms, coords)
        for basis in (bs1, bs2):
            mol = gto.Mole()
            mol.atom = xyz_str
            mol.basis = basis
            mol.spin = int(spin)
            mol.charge = 0
            mol.nthread = pyscf_threads
            mol.max_memory = 8000
            mol.build()
            mf = scf.RHF(mol)
            mf.max_memory = 14330
            mf.conv_tol = 1e-9
            mf.max_cycle = 100
            scf_e = mf.kernel()
            pyscf_energies[basis] = float(scf_e)
    except Exception as e:
        if sys.modules.get('__main__'):  # Only print if not silent
            pass
        pyscf_energies = {bs1: None, bs2: None}
    return pyscf_energies


# -------------------------
# 3-point parabolic helper (robust)
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


# ---- CSV WRITER FUNCTION: Write cycle energies with PySCF data ----
def write_cycle_energies_with_pyscf(outputs_dir, prefix, history, basis1, basis2):
    """
    Write cycle energies to CSV with both PySCF (HF-SCF) and CBS energies.

    Output CSV has columns:
    Cycle, CBS_Energy, PySCF_HF_<basis1>, PySCF_HF_<basis2>, Energy_Difference
    """
    csv_file = outputs_dir / f"{prefix}_cycles_with_pyscf.csv"

    with open(csv_file, 'w') as f:
        # Header
        f.write(f"Cycle,CBS_Energy_Ha,PySCF_HF_{basis1}_Ha,PySCF_HF_{basis2}_Ha,ΔE_from_initial_Ha,ΔE_from_previous_Ha\n")

        initial_energy = history[0]['energy'] if history else 0.0
        previous_energy = initial_energy

        for entry in history:
            cycle = entry['cycle']
            cbs_e = entry['energy']
            pyscf_bs1 = entry.get('pyscf_hf_bs1', None)
            pyscf_bs2 = entry.get('pyscf_hf_bs2', None)

            delta_from_initial = cbs_e - initial_energy
            delta_from_previous = cbs_e - previous_energy

            # Format: use 'N/A' if PySCF energies not available
            pyscf_bs1_str = f"{pyscf_bs1:.10f}" if pyscf_bs1 is not None else "N/A"
            pyscf_bs2_str = f"{pyscf_bs2:.10f}" if pyscf_bs2 is not None else "N/A"

            f.write(
                f"{cycle},{cbs_e:.10f},{pyscf_bs1_str},{pyscf_bs2_str},{delta_from_initial:.10e},{delta_from_previous:.10e}\n")

            previous_energy = cbs_e

    return csv_file


# -------------------------
# Main optimization loop (SQM-style) — ALL-PER-CYCLE approach
# -------------------------
def optimize_from_xyz(atoms, coords, method=DEFAULT_METHOD, maxcycle=MAXCYCLE_DEFAULT, fac_mult=FAC_DEFAULT,
                      x1=X1_DEFAULT, x2=X2_DEFAULT, x1_hf=X1HF_DEFAULT, x2_hf=X2HF_DEFAULT, beta=BETA_DEFAULT,
                      basis_pair=None, spin: int = 0, debug: bool = False, energy_accept_tol: float | None = None,
                      workers: int = 1, outputs_dir: Path | None = None, base_name: str = "geometry"):
    """
    ALL-PER-CYCLE OPTIMIZATION STRATEGY
    ====================================

    For each optimization cycle:
      1. Evaluate ALL internal coordinates (3-point sampling + parabolic fit)
      2. For EACH internal:
         - If predicted energy improves beyond threshold:
           * Apply the change to get new coordinates
           * Validate by calculating actual CBS energy
           * If confirmed: keep the new geometry
           * If not confirmed: try next internal
      3. Regenerate internal coordinates (accounts for coupling)
      4. Reduce displacement factor for next cycle

    This allows multiple coordinate updates per cycle (unlike single-best),
    but validates each one independently.
    """
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

    # Track per-cycle geometries
    cycle_geometries = {}
    if outputs_dir:
        outputs_dir = Path(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        # Save initial geometry
        initial_xyz = outputs_dir / f"{base_name}_cycle_000_initial.xyz"
        write_xyz(initial_xyz, atoms, current_coords, f"Initial geometry - cycle 0")
        cycle_geometries[0] = {
            "path": str(initial_xyz),
            "atoms": atoms,
            "coords": current_coords.copy(),
            "energy": None,
            "description": "Initial"
        }

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
        cur_xyz = xyz_to_pyscf_string(atoms, current_coords)
        try:
            current_energy = compute_cbs_energy_from_xyz_cached(cur_xyz, method, a_corr, b_hf, bs1, bs2, spin)
        except Exception as e:
            raise RuntimeError(f"CBS evaluation at cycle start failed: {e}")

        print(
            f"\n>>> Cycle {cycle}/{maxcycle}, displacement_factor={displacement_factor:.6f}, current E = {current_energy:.10f} Ha")

        # ---- Evaluate ALL internals and update each one that improves energy ----
        def evaluate_internal(ic):
            """
            Evaluate a single internal coordinate:
            1. Sample 3 points: -h, 0, +h
            2. Fit parabola
            3. Return predicted minimum and validation coords
            """
            tp = ic['type']
            inds = ic['inds']

            if tp == 'bond':
                base = ic.get('value', _value_for_internal(tp, inds, current_coords))
                h = displacement_factor * base
                ds = np.array([-h, 0.0, h], dtype=float)
            elif tp in ('angle', 'dihedral'):
                base = ic.get('value', _value_for_internal(tp, inds, current_coords))
                h = 2.0 * (displacement_factor * 100.0)
                ds = np.array([-h, 0.0, h], dtype=float)
            else:
                ds = np.array([0.0])
                h = 0.0

            es = []
            coords_list = []
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
                    es.append(float(E))
                    coords_list.append(new_coords)
                except Exception:
                    es.append(float('inf'))
                    coords_list.append(None)
                    if debug:
                        print(f"    eval failed for IC {tp} {inds} displacement {d}")

            es = np.array(es, dtype=float)
            if np.all(np.isinf(es)):
                return None

            x_min_disp, e_min = parabolic_minimum_3pt(ds, es)

            # Calculate curvature (second derivative)
            grad0 = None
            curvature = None
            try:
                Eminus = es[0]
                E0 = es[1]
                Eplus = es[2]
                grad0 = (Eplus - Eminus) / (2.0 * (ds[2] - ds[1]))
                curvature = (Eplus + Eminus - 2.0 * E0) / ((ds[2] - ds[1]) ** 2)
            except Exception:
                pass

            deltaE_pred = current_energy - e_min

            # Try to use predicted minimum if curvature is good
            if curvature is not None and curvature > MIN_CURVATURE and deltaE_pred > ENERGY_ACCEPT_TOL:
                try:
                    if tp == 'bond':
                        i, j = inds
                        coords_at_pred = apply_bond_change(current_coords, i, j, ic.get('value',
                                                                                        _value_for_internal(tp, inds,
                                                                                                            current_coords)) + x_min_disp,
                                                           atoms)
                    elif tp == 'angle':
                        i, j, k = inds
                        coords_at_pred = apply_angle_change(current_coords, i, j, k, ic.get('value',
                                                                                            _value_for_internal(tp,
                                                                                                                inds,
                                                                                                                current_coords)) + x_min_disp)
                    elif tp == 'dihedral':
                        i, j, k, l = inds
                        coords_at_pred = apply_dihedral_change(current_coords, i, j, k, l, ic.get('value',
                                                                                                  _value_for_internal(
                                                                                                      tp, inds,
                                                                                                      current_coords)) + x_min_disp)
                    else:
                        coords_at_pred = current_coords.copy()

                    xyz_pred = xyz_to_pyscf_string(atoms, coords_at_pred)
                    E_pred_true = compute_cbs_energy_from_xyz_cached(xyz_pred, method, a_corr, b_hf, bs1, bs2, spin)
                    deltaE_pred_true = current_energy - float(E_pred_true)
                except Exception:
                    E_pred_true = float('inf')
                    deltaE_pred_true = -1.0

                if E_pred_true != float('inf') and deltaE_pred_true > ENERGY_ACCEPT_TOL:
                    return (deltaE_pred_true, x_min_disp, coords_at_pred, ic, curvature, grad0, 'predicted')

            # Fallback: use sampled best
            idx_best = int(np.nanargmin(es))
            sampled_best_coords = coords_list[idx_best]
            sampled_best_energy = float(es[idx_best]) if not np.isinf(es[idx_best]) else None

            if sampled_best_energy is not None and (current_energy - sampled_best_energy) > ENERGY_ACCEPT_TOL:
                return ((current_energy - sampled_best_energy), ds[idx_best], sampled_best_coords, ic, curvature, grad0,
                        'sampled')

            return None

        # Evaluate all internals (in parallel or sequential)
        candidates = []
        if workers is None or workers <= 1:
            for ic in internals:
                try:
                    res = evaluate_internal(ic)
                    if res is not None:
                        candidates.append(res)
                except Exception:
                    if debug:
                        print("  evaluator error for internal", ic.get('inds'))
        else:
            with ThreadPoolExecutor(max_workers=workers) as exc:
                future_map = {exc.submit(evaluate_internal, ic): ic for ic in internals}
                for fut in as_completed(future_map):
                    try:
                        res = fut.result()
                        if res is not None:
                            candidates.append(res)
                    except Exception as e:
                        if debug:
                            print("  worker error:", e)

        # ---- ALL-PER-CYCLE: Apply EACH profitable candidate ----
        num_applied = 0
        if candidates:
            # Sort by predicted energy improvement (largest first)
            candidates.sort(key=lambda x: x[0], reverse=True)

            for deltaE, chosen_disp, chosen_coords, chosen_ic, curvature, grad0, which in candidates:
                # Update current geometry
                current_coords = chosen_coords.copy()

                # Regenerate internals (critical for accounting for coupling)
                internals = generate_internals_from_geometry(atoms, current_coords)

                # Recalculate energy at this new geometry
                try:
                    cur_xyz_after = xyz_to_pyscf_string(atoms, current_coords)
                    actual_E_after = compute_cbs_energy_from_xyz_cached(cur_xyz_after, method, a_corr, b_hf, bs1, bs2,
                                                                        spin)
                except Exception:
                    actual_E_after = float('inf')

                print(f"  ✓ Applied: {_label_internal(chosen_ic['type'], chosen_ic['inds'], atoms)}")
                print(f"    type={which}, disp={chosen_disp:.6f}, ΔE_pred={deltaE:.3e}, E_after={actual_E_after:.10f}")

                current_energy = float(actual_E_after)
                num_applied += 1
        else:
            print(f"  No acceptable candidates this cycle.")

        # Calculate PySCF energies for CSV logging
        pyscf_energies = compute_pyscf_hf_energies(atoms, current_coords, bs1, bs2, spin, PYSCF_THREADS)

        history.append({
            'cycle': cycle,
            'energy': float(current_energy),
            'pyscf_hf_bs1': pyscf_energies.get(bs1, None),
            'pyscf_hf_bs2': pyscf_energies.get(bs2, None)
        })
        cyc_map = {}
        for lbl, tp, inds in zip(baseline_labels, baseline_types, baseline_inds):
            cyc_map[lbl] = _value_for_internal(tp, inds, current_coords)
        internals_trace.append(cyc_map)

        # Save per-cycle geometry
        if outputs_dir:
            cycle_xyz = outputs_dir / f"{base_name}_cycle_{cycle:03d}.xyz"
            write_xyz(cycle_xyz, atoms, current_coords,
                      f"Cycle {cycle} - Energy: {current_energy:.10f} Ha - {num_applied} updates")
            cycle_geometries[cycle] = {
                "path": str(cycle_xyz),
                "atoms": atoms,
                "coords": current_coords.copy(),
                "energy": float(current_energy),
                "description": f"Cycle {cycle} ({num_applied} updates)"
            }

        if cycle > 1:
            ediff = abs(history[-1]['energy'] - history[-2]['energy'])
            print(f"  ΔE since last cycle: {ediff:.4e} Ha")
            if ediff < ENERGY_CRIT:
                print("Converged by energy criterion")
                converged = True
                break

        displacement_factor *= CUT

    return atoms, current_coords, history, converged, baseline_labels, internals_trace, cycle_geometries


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

    atoms_out, coords_out, history, converged, baseline_labels, internals_trace, cycle_geometries = optimize_from_xyz(
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
        workers=workers,
        outputs_dir=outputs_dir,
        base_name=Path(input_xyz).stem
    )

    base = Path(input_xyz).stem
    prefix = f"{base}_SQM"
    cycles_file = write_cycle_energies(outputs_dir, prefix, history)
    xyz_file = write_final_xyz(outputs_dir, prefix, atoms_out, coords_out, history[-1]['energy'] if history else 0.0)

    # Write enhanced CSV with PySCF + CBS energies
    if basis_pair is None:
        basis_pair = (basis_sets[0], basis_sets[1])
    enhanced_cycles_file = write_cycle_energies_with_pyscf(outputs_dir, prefix, history, basis_pair[0], basis_pair[1])

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

    # Print summary of saved geometries
    print("\n" + "=" * 80)
    print("SAVED GEOMETRIES (per-cycle XYZ files):")
    print("=" * 80)
    for cycle_num in sorted(cycle_geometries.keys()):
        geo_data = cycle_geometries[cycle_num]
        energy_str = f"{geo_data['energy']:.10f} Ha" if geo_data['energy'] is not None else "N/A"
        print(f"  Cycle {cycle_num:3d}: {Path(geo_data['path']).name:40s} E = {energy_str}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("ENERGY CSV FILES:")
    print("=" * 80)
    print(f"  Standard CSV: {Path(cycles_file).name}")
    print(f"  Enhanced CSV (with PySCF): {Path(enhanced_cycles_file).name}")
    print("=" * 80)

    return {
        "history": history,
        "final_energy": float(history[-1]['energy']) if history else None,
        "final_cart": coords_out,
        "symbols": atoms_out,
        "outputs": {"cycles": str(cycles_file), "cycles_with_pyscf": str(enhanced_cycles_file), "xyz": str(xyz_file)},
        "converged": converged,
        "internals_trace": {"labels": baseline_labels, "trace": internals_trace},
        "cycle_geometries": cycle_geometries,
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
    p.add_argument("--workers", type=int, default=1, help="Number of parallel internal evaluators (default 1)")
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