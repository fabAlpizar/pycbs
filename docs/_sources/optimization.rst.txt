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


Frozen-core option
------------------

By default, MP2  and CCSD(T) calculations correlate all electrons in all available orbitals. To
reduce computational cost while preserving chemically relevant correlation, this
package supports a frozen-core approximation following the same conventions used
in PySCF.

Freezing core orbitals is often a reliable approximation because core electrons
usually contribute very little to correlation energy differences. Therefore,
excluding them from the correlation treatment significantly reduces computational
time while maintaining good accuracy for most chemical applications.

Usage notes
~~~~~~~~~~~

- The ``frozen`` keyword accepts either:

  - an integer ``n`` — freezes the ``n`` lowest-energy occupied orbitals
    (typically core orbitals).

  - a list of **0-based orbital indices** to freeze specific orbitals, allowing
    fine control over which occupied or virtual orbitals are excluded from the
    correlation calculation.

- Orbital indexing follows the standard **0-based indexing** convention used in Python.

Examples
~~~~~~~~
It is shown the MP2 example but it works as well with CCSD(T) method.

.. code-block:: python

    # freeze 2 core orbitals
    mymp = mp.MP2(mf, frozen=2).run()

    # freeze specific orbitals using indices
    mymp = mp.MP2(mf, frozen=[0,1]).run()

    # freeze 2 core orbitals and 3 virtual orbitals
    mymp = mp.MP2(mf, frozen=[0,1,16,17,18]).run()

Automatic core detection
~~~~~~~~~~~~~~~~~~~~~~~~

The number of orbitals to freeze can also be determined automatically using
PySCF's internal rule:

.. code-block:: python

    mymp = mp.MP2(mf).set_frozen().run()

The ``set_frozen()`` method automatically freezes the core orbitals based on
the total number of core electrons for each atom in the molecule. The current
rule follows the same convention used by the ORCA quantum chemistry program.

When the ``frozen`` option is provided in the configuration file, the value is
passed directly to the correlated electronic structure calculation
(MP2 or CCSD(T)). Therefore, the behavior is identical to the native PySCF
implementation.

If ``frozen`` is not specified, all orbitals are correlated by default.

Remarks
-------

We strongly recommend using a **specific and descriptive filename** for the
input ``.xyz`` geometry used in the optimization.

The program automatically generates output filenames by taking the base name
of the input file and appending the optimizer used in the calculation
(``sqm`` or ``lbfgs``). For example:

- Input geometry::

    molecule_opt.xyz

- Output files::

    molecule_lbfgs_final_opt.xyz
    molecule_sqm_final_opt.xyz
    molecule_lbfgs_cycles_energies.csv
    molecule_sqm_cycles_energies.csv

Because of this behavior, using clear and unique input filenames helps avoid
confusion or accidental overwriting when running multiple optimizations.

A recommended naming convention is::

    <system>_<method>.xyz

Example::

    ethanol_ccsdt.xyz


