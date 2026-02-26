Geometrical optimization
========================
Overview
--------
This package provides a high-level Python implementation for geometry optimization
using Complete Basis Set (CBS) extrapolation with PySCF. It supports **CCSD(T)** and
**MP2** methods, generates redundant internal coordinates (RICs), evaluates
all internal coordinates per optimization cycle, and applies energy-validated
updates to molecular geometries.


Gradient based
~~~~~~~~~~~~~~
.. code-block:: ini

    [optimization3]
    optimization = True            ; Enable optimization (True/False), default: True
    input_xyz = path/to/xyz/file   ; Path to input XYZ geometry file, no default
    method = CCSD(T)               ; Method for energy evaluation ('CCSD(T)' or 'MP2'), default: CCSD(T)
    optimizer = L-BFGS-B           ; Optimization algorithm ('L-BFGS-B', 'sqm'), default: L-BFGS-B
    basis1 = cc-pvdz               ; First basis set for CBS extrapolation, default: cc-pvdz
    basis2 = cc-pvtz               ; Second basis set for CBS extrapolation, default: cc-pvtz
    spin = 0                       ; Spin multiplicity (0=singlet, 1=doublet, etc.), default: 0
    x1 = 1.852                     ; CBS extrapolation coefficient a_corr or first basis distance, default: 1.852
    x2 = 2.639                     ; CBS extrapolation coefficient b_hf or second basis distance, default: 2.639
    x1hf = 3.02                    ; HF exponential factor for basis1, default: 3.02
    x2hf = 3.64                    ; HF exponential factor for basis2, default: 3.64
    beta = 1.0                     ; HF exponential scaling factor, default: 1.0
    maxcycle = 100                 ; Maximum number of optimization cycles, default: 100
    fac_mult = 0.05                ; Displacement factor multiplier for internal coordinates, default: 0.05
    energy_crit = 1e-8             ; Energy convergence criterion in Hartree, default: 1e-8
    cut = 0.75                     ; Per-cycle reduction factor for displacement, default: 0.75
    workers = 1                    ; Number of parallel threads for PySCF evaluations, default: 1
    debug = False                  ; Print detailed per-cycle information, default: False
    output_dir = ./results         ; Directory to store cycle geometries and CSV, default: ./results

SQM
~~~
.. code-block:: ini

    [optimization3]
    optimization = True            ; Enable optimization (True/False), default: True
    input_xyz = path/to/xyz/file   ; Path to input XYZ geometry file, no default
    method = CCSD(T)               ; Method for energy evaluation ('CCSD(T)' or 'MP2'), default: CCSD(T)
    optimizer = sqm                ; Optimization algorithm ('sqm' for internal coordinates), default: sqm
    basis1 = cc-pvdz               ; First basis set for CBS extrapolation, default: cc-pvdz
    basis2 = cc-pvtz               ; Second basis set for CBS extrapolation, default: cc-pvtz
    spin = 0                       ; Spin multiplicity (0=singlet, 1=doublet, etc.), default: 0
    x1 = 1.852                     ; CBS extrapolation coefficient a_corr or first basis distance, default: 1.852
    x2 = 2.639                     ; CBS extrapolation coefficient b_hf or second basis distance, default: 2.639
    x1hf = 3.02                    ; HF exponential factor for basis1, default: 3.02
    x2hf = 3.64                    ; HF exponential factor for basis2, default: 3.64
    beta = 1.0                     ; HF exponential scaling factor, default: 1.0
    maxcycle = 100                 ; Maximum number of optimization cycles, default: 100
    fac_mult = 0.05                ; Displacement factor multiplier for internal coordinates, default: 0.05
    energy_crit = 1e-8             ; Energy convergence criterion in Hartree, default: 1e-8
    cut = 0.75                     ; Per-cycle reduction factor for displacement, default: 0.75
    workers = 1                    ; Number of parallel threads for PySCF evaluations, default: 1
    debug = False                  ; Print detailed per-cycle information, default: False
    output_dir = ./results         ; Directory to store cycle geometries and CSV, default: ./results
    energy_accept_tol = 1e-6       ; Minimum energy improvement to accept step, default: 1e-6
    geom_rebuild = True             ; Regenerate internal coordinates each cycle, default: True
    label_internals = True          ; Include human-readable labels for bonds/angles/dihedrals, default: True