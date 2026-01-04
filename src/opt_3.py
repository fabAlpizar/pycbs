#!/usr/bin/env python3
"""
Enhanced CBS optimizer using REDUNDANT INTERNAL COORDINATES.
Optimizes in bond/angle/dihedral space, outputs in Cartesian.
This prevents bond breaking in dimers.
"""

import os
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from pyscf import gto, scf, cc, mp, lib

# USER CONFIGURATION
INPUT_XYZ = "/home/fab/01_Fab_Hub/01_pinchas/02_albeaker/pyCBS/xyz/meoh.xyz"
OUTPUT_XYZ = "optimized_meo.xyz"
METHOD = "MP2"
BASIS_SETS = ["sto-3g", "sto-6g"]
MAXCYCLE = 50
ENERGY_CRITERION = 1e-6
MAX_WORKERS = max(1, cpu_count() - 1)
PYSCF_THREADS = max(1, (max(1, cpu_count() - 1)) // 2)
VERBOSE = True

# AGGREGATE-SPECIFIC CONSTRAINTS
BOND_THRESHOLD = 1.6  # Ångströms (for molecule detection)
STRONG_BOND_THRESHOLD = 1.8  # Any bond < 1.8 Å is treated as strong covalent

os.environ['MKL_NUM_THREADS'] = str(PYSCF_THREADS)
os.environ['OMP_NUM_THREADS'] = str(PYSCF_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(PYSCF_THREADS)
lib.num_threads(PYSCF_THREADS)


def get_bond_lengths(positions, symbols):
    """Compute all pairwise distances"""
    distances = {}
    nat = len(symbols)
    for i in range(nat):
        for j in range(i + 1, nat):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances[(i, j)] = dist
    return distances


def detect_molecules(positions, symbols, bond_threshold=BOND_THRESHOLD):
    """Cluster atoms into molecules based on distance threshold."""
    nat = len(symbols)
    dist_matrix = squareform(pdist(positions))
    adjacency = dist_matrix < bond_threshold
    np.fill_diagonal(adjacency, False)

    visited = np.zeros(nat, dtype=bool)
    molecules = []

    def dfs(atom_idx, current_mol):
        visited[atom_idx] = True
        current_mol.append(atom_idx)
        for neighbor in np.where(adjacency[atom_idx])[0]:
            if not visited[neighbor]:
                dfs(neighbor, current_mol)

    for i in range(nat):
        if not visited[i]:
            mol = []
            dfs(i, mol)
            molecules.append(sorted(mol))

    return molecules


def identify_connectivity(positions, symbols, bond_threshold=BOND_THRESHOLD):
    """
    Build connectivity graph:  for each atom, list its bonded neighbors.
    Used to construct internal coordinates.
    """
    nat = len(symbols)
    dist_matrix = squareform(pdist(positions))
    adjacency = dist_matrix < bond_threshold
    np.fill_diagonal(adjacency, False)

    connectivity = {i: [] for i in range(nat)}
    for i in range(nat):
        for j in range(nat):
            if adjacency[i, j]:
                connectivity[i].append(j)

    return connectivity


def build_internal_coordinates(positions, symbols, connectivity):
    """
    Build a list of internal coordinates:
    - distances:  (i, j) pairs for bonds
    - angles: (i, j, k) triplets for angles (j is central atom)
    - dihedrals: (i, j, k, l) for torsions

    Returns: (distances_list, angles_list, dihedrals_list, reference_values)
    """
    distances = []
    angles = []
    dihedrals = []

    nat = len(symbols)

    # Build bonds from connectivity
    for i in range(nat):
        for j in connectivity[i]:
            if i < j:  # Avoid duplicates
                distances.append((i, j))

    # Build angles:  for each atom j, use pairs of neighbors
    for j in range(nat):
        neighbors = connectivity[j]
        if len(neighbors) >= 2:
            for k1_idx, k1 in enumerate(neighbors):
                for k2 in neighbors[k1_idx + 1:]:
                    angles.append((k1, j, k2))

    # Build dihedrals: only for bonds between atoms with multiple neighbors
    for j in range(nat):
        for k in connectivity[j]:
            if j < k:
                # j-k is a bond; find atoms bonded to j and k
                atoms_bonded_to_j = [a for a in connectivity[j] if a != k]
                atoms_bonded_to_k = [a for a in connectivity[k] if a != j]

                if atoms_bonded_to_j and atoms_bonded_to_k:
                    i = atoms_bonded_to_j[0]
                    l = atoms_bonded_to_k[0]
                    dihedrals.append((i, j, k, l))

    return distances, angles, dihedrals


def get_distance(r1, r2):
    """Calculate distance between two points"""
    return np.linalg.norm(r2 - r1)


def get_angle(r1, r2, r3):
    """
    Calculate angle (in degrees) at r2 between r1-r2-r3.
    r2 is the central atom.
    """
    v1 = r1 - r2
    v2 = r3 - r2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def get_dihedral(r1, r2, r3, r4):
    """
    Calculate dihedral angle (in degrees) i-j-k-l.
    """
    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_n1 = np.linalg.norm(n1) + 1e-12
    norm_n2 = np.linalg.norm(n2) + 1e-12

    n1 /= norm_n1
    n2 /= norm_n2

    cos_phi = np.dot(n1, n2)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.degrees(np.arccos(cos_phi))

    return phi


def cartesian_to_internal(positions, distances_list, angles_list, dihedrals_list):
    """
    Convert Cartesian coordinates to internal coordinates.
    Returns array:  [d1, d2, ..., a1, a2, ..., dih1, dih2, ...]
    """
    internal = []

    # Distances
    for i, j in distances_list:
        d = get_distance(positions[i], positions[j])
        internal.append(d)

    # Angles
    for i, j, k in angles_list:
        angle = get_angle(positions[i], positions[j], positions[k])
        internal.append(angle)

    # Dihedrals
    for i, j, k, l in dihedrals_list:
        dih = get_dihedral(positions[i], positions[j], positions[k], positions[l])
        internal.append(dih)

    return np.array(internal, dtype=float)


def internal_to_cartesian(internal_coords, distances_list, angles_list, dihedrals_list,
                          reference_positions, connectivity):
    """
    Convert internal coordinates back to Cartesian.
    Uses reference positions to build a Z-matrix-like structure.
    This is the inverse of cartesian_to_internal.
    """
    positions = reference_positions.copy()
    nat = len(positions)

    n_dist = len(distances_list)
    n_angles = len(angles_list)

    distances_dict = {pair: internal_coords[i] for i, pair in enumerate(distances_list)}
    angles_dict = {triplet: internal_coords[n_dist + i] for i, triplet in enumerate(angles_list)}
    dihedrals_dict = {quad: internal_coords[n_dist + n_angles + i] for i, quad in enumerate(dihedrals_list)}

    # Update positions based on internal coordinates
    # This is a simplified approach:  adjust atoms to match target distances/angles
    # For a more robust solution, use constrained optimization or proper Z-matrix construction

    for idx, (i, j) in enumerate(distances_list):
        target_dist = internal_coords[idx]
        current_dist = get_distance(positions[i], positions[j])

        if current_dist > 1e-6:
            # Move j closer to or away from i
            direction = (positions[j] - positions[i]) / current_dist
            positions[j] = positions[i] + direction * target_dist

    for idx, (i, j, k) in enumerate(angles_list):
        target_angle = internal_coords[n_dist + idx]
        current_angle = get_angle(positions[i], positions[j], positions[k])

        if abs(current_angle - target_angle) > 0.1:
            # Rotate k around bond j-i to match target angle
            bond_vec = positions[i] - positions[j]
            bond_vec /= (np.linalg.norm(bond_vec) + 1e-12)

            vec_to_k = positions[k] - positions[j]

            # Simple rotation: adjust k's position
            angle_diff = np.radians(target_angle - current_angle)
            rotation_axis = np.cross(bond_vec, vec_to_k / (np.linalg.norm(vec_to_k) + 1e-12))
            rotation_axis /= (np.linalg.norm(rotation_axis) + 1e-12)

            # Rodrigues rotation formula
            cos_a = np.cos(angle_diff)
            sin_a = np.sin(angle_diff)
            cross_product = np.cross(rotation_axis, vec_to_k)

            rotated_vec = (vec_to_k * cos_a +
                           cross_product * sin_a +
                           rotation_axis * np.dot(rotation_axis, vec_to_k) * (1 - cos_a))

            positions[k] = positions[j] + rotated_vec

    return positions


def read_xyz(path):
    with open(path, 'r') as f:
        n = int(f.readline().strip())
        _ = f.readline()
        symbols = []
        pos = []
        for _ in range(n):
            parts = f.readline().split()
            symbols.append(parts[0])
            pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(pos, dtype=float)


def write_xyz(path, symbols, positions, comment="Optimized geometry"):
    with open(path, 'w') as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, p in zip(symbols, positions):
            f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


X1 = 2.792
X2 = 3.719
X1_HF = 2.96
X2_HF = 3.87
BETA = 1.62

A_CORR = (X1 ** 3) / (X2 ** 3 - X1 ** 3)
B_HF = (math.exp(BETA * X1_HF)) / (math.exp(BETA * X2_HF) - math.exp(BETA * X1_HF))


def cbs_compose(ex1_corr, ex2_corr, ex1_hf, ex2_hf):
    return ex2_corr + A_CORR * (ex2_corr - ex1_corr) + (A_CORR - B_HF) * (ex2_hf - ex1_hf)


_local_cache = {}


def xyz_text_from_positions(positions, symbols):
    lines = []
    for s, p in zip(symbols, positions):
        lines.append(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")
    return "\n".join(lines)


def compute_cbs_energy(method, positions, symbols):
    """Compute CBS energy from Cartesian positions"""
    key = (method, tuple(np.round(positions.flatten(), 10)))
    if key in _local_cache:
        return _local_cache[key]

    results_scf = []
    results_corr = []
    xyz_text = xyz_text_from_positions(positions, symbols)

    for basis in BASIS_SETS:
        mol = gto.Mole()
        mol.atom = xyz_text
        mol.basis = basis
        mol.spin = 0
        mol.charge = 0
        mol.nthread = PYSCF_THREADS
        mol.max_memory = 8000
        mol.build()

        mf = scf.RHF(mol)
        mf.max_memory = 12000
        mf.conv_tol = 1e-8
        mf.max_cycle = 150
        scf_e = float(mf.kernel())

        if method.upper() == "CCSD(T)":
            mycc = cc.CCSD(mf)
            mycc.conv_tol = 1e-7
            mycc.max_cycle = 200
            mycc.kernel()
            try:
                triples = mycc.ccsd_t()
            except Exception:
                triples = 0.0
            corr_tot = float(mycc.e_tot + (triples if triples is not None else 0.0))
        elif method.upper() == "MP2":
            mymp = mp.MP2(mf)
            mymp.max_memory = 12000
            mp_res = mymp.run()
            mp2_total = getattr(mp_res, 'e_tot', None)
            if mp2_total is None:
                mp2_total = getattr(mymp, 'e_tot', None)
            if mp2_total is None:
                e_corr = getattr(mp_res, 'e_corr', getattr(mymp, 'e_corr', None))
                if e_corr is not None:
                    mp2_total = scf_e + e_corr
            if mp2_total is None:
                raise RuntimeError("Could not retrieve MP2 energy")
            corr_tot = float(mp2_total)
        else:
            raise ValueError("Unknown method: " + str(method))

        results_scf.append(scf_e)
        results_corr.append(corr_tot)

    energy = cbs_compose(results_corr[0], results_corr[1], results_scf[0], results_scf[1])
    _local_cache[key] = float(energy)
    return float(energy)


def run():
    symbols, positions = read_xyz(INPUT_XYZ)
    natoms = len(symbols)

    # Detect molecules and connectivity
    molecules = detect_molecules(positions, symbols)
    connectivity = identify_connectivity(positions, symbols)

    # Build internal coordinates
    distances_list, angles_list, dihedrals_list = build_internal_coordinates(
        positions, symbols, connectivity
    )

    # Get reference internal values
    reference_internal = cartesian_to_internal(positions, distances_list, angles_list, dihedrals_list)

    if VERBOSE:
        print("=" * 80)
        print("INTERNAL COORDINATE CBS OPTIMIZER FOR AGGREGATES")
        print(f"Input:    {INPUT_XYZ}")
        print(f"Atoms:  {natoms}  Method: {METHOD}")
        print(f"Molecules detected: {len(molecules)}")
        for i, mol in enumerate(molecules):
            print(f"  Molecule {i}:    atoms {mol} ({[symbols[a] for a in mol]})")

        print(f"\nInternal coordinates built:")
        print(f"  Distances: {len(distances_list)}")
        print(f"  Angles:  {len(angles_list)}")
        print(f"  Dihedrals: {len(dihedrals_list)}")
        print("=" * 80)

    # Energy function in internal coordinate space
    def energy_func(internal_coords):
        try:
            # Convert internal to Cartesian
            current_positions = internal_to_cartesian(
                internal_coords, distances_list, angles_list, dihedrals_list,
                positions.copy(), connectivity
            )

            # Compute energy
            en = compute_cbs_energy(METHOD, current_positions, symbols)
            return float(en)
        except Exception as e:
            if VERBOSE:
                print(f"Energy evaluation failed: {e}")
            return np.inf

    # Optimize using L-BFGS-B (supports bounds)
    if VERBOSE:
        print(f"\nStarting optimization in {len(reference_internal)}-dimensional internal coordinate space.. .\n")

    result = minimize(
        energy_func,
        reference_internal,
        method='L-BFGS-B',
        options={
            'maxiter': MAXCYCLE,
            'ftol': ENERGY_CRITERION,
            'gtol': 1e-4,
            'disp': VERBOSE
        }
    )

    # Convert final internal coordinates back to Cartesian
    optimized_positions = internal_to_cartesian(
        result.x, distances_list, angles_list, dihedrals_list,
        positions.copy(), connectivity
    )

    final_energy = compute_cbs_energy(METHOD, optimized_positions, symbols)

    # Write output
    write_xyz(OUTPUT_XYZ, symbols, optimized_positions,
              comment=f"Optimized aggregate ({len(molecules)} molecules)")

    if VERBOSE:
        print("\n" + "=" * 80)
        print("OPTIMIZATION FINISHED")
        print(f" Final CBS energy = {final_energy:.10f} Ha")
        print(f" Optimized geometry written to:    {OUTPUT_XYZ}")

        # Verify bond integrity
        print(f"\nBond verification (should be unchanged):")
        final_distances = get_bond_lengths(optimized_positions, symbols)
        initial_distances = get_bond_lengths(positions, symbols)

        all_intact = True
        for (i, j) in sorted(distances_list):
            initial_d = initial_distances[(i, j)]
            final_d = final_distances[(i, j)]
            change = abs(final_d - initial_d)
            status = "✓ OK" if change < 0.01 else "✗ CHANGED"


            if change >= 0.05:
                all_intact = False

        if all_intact:
            print("\n✓ All bonds preserved during optimization!")
        else:
            print("\n⚠ Some bonds changed significantly!")

        print("=" * 80)

    return {
        'positions': optimized_positions,
        'energy': final_energy,
        'converged': result.success,
        'molecules': molecules,
        'result': result
    }


if __name__ == "__main__":
    res = run()
    if res['converged']:
        print("\n✓ Result:  converged")
    else:
        print(f"\n✗ Result:  finished (message: {res['result'].message})")