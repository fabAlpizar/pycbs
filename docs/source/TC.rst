Theoretical Background
======================

.. _theoretical_background:

Overview
--------

This section summarizes the theoretical framework underlying Complete Basis Set (CBS)
extrapolation and CBS-driven geometry optimization as implemented in PyCBS.

The objective is to model basis-set convergence using physically motivated asymptotic laws
and to obtain reliable CBS-limit estimates for energies, properties, and geometries with
controlled computational cost.


Energy Partitioning
-------------------

CBS extrapolation is performed by separating the total electronic energy into
Hartree–Fock (HF) and correlation components:

.. math::

   E_X = E_X^{\mathrm{HF}} + E_X^{\mathrm{cor}}

Because these components exhibit distinct convergence behaviors with respect to
basis-set cardinal number :math:`X`, they are extrapolated independently.


Hartree–Fock Convergence
------------------------

HF energies converge exponentially toward the CBS limit:

.. math::

   E_X^{\mathrm{HF}} = E_\infty^{\mathrm{HF}} + B e^{-\beta X}

Exponential models provide superior stability compared to inverse-power forms
for the mean-field component.

Optimized exponential variants generally provide improved numerical stability
and reduced systematic bias.


Correlation Energy Convergence
------------------------------

Correlation energies follow an inverse power-law asymptotic behavior:

.. math::

   E_X^{\mathrm{cor}} = E_\infty^{\mathrm{cor}} + \frac{A}{X^\alpha}

For large :math:`X`, the dominant term approaches :math:`X^{-3}`.

The program includes multiple inverse-power and unified extrapolation schemes
designed to improve robustness when high-cardinality basis sets are not available.


Hierarchical Numbers
--------------------

Instead of raw cardinal numbers (2, 3, 4, 5),
optimized hierarchical (effective) numbers may be employed:

.. math::

   X \rightarrow \tilde{X}

This reparameterization improves linearity in inverse-power fits and enhances
extrapolation stability.

Hierarchical mappings should remain consistent across extrapolation schemes
unless explicitly re-optimized.


CBS for Molecular Properties
----------------------------

Properties such as polarizabilities and vibrational frequencies may be partitioned
analogously:

.. math::

   \zeta = \zeta^{\mathrm{HF}} + \zeta^{\mathrm{cor}}

Assuming similar convergence trends as energies, inverse-power extrapolation
can be applied to the correlation contribution. Deviations from strict
asymptotic behavior should be evaluated carefully.


CBS Geometry Optimization
-------------------------

Geometry optimization at the CBS limit is performed directly on the extrapolated
energy surface.

Two strategies are supported:

- Successive Quadratic Minimization (SQM)
- L-BFGS-B gradient-based optimization

At each optimization step:

1. Energies are computed at two basis levels.
2. HF and correlation components are extrapolated.
3. The CBS total energy is used as the objective function.

This ensures structural refinement on a consistent CBS surface.


Practical Recommendations
-------------------------

For most applications, the best accuracy-to-cost ratio is obtained with
MP2-based extrapolations:

- MP2/VDZ–VTZ
- MP2/VTZ–VQZ

These combinations provide robust CBS estimates with moderate computational cost.

For higher precision, especially in thermochemistry or structural benchmarks,
CCSD(T)-based extrapolation is recommended.

Although CCSD(T) provides superior accuracy, its computational scaling
limits applicability to small and medium-sized systems.


Error Sources
-------------

CBS extrapolation errors may arise from:

- Incomplete asymptotic regime (small basis sets)
- Imbalance between HF and correlation treatment
- Numerical instability in nonlinear fitting
- Inadequate treatment of higher-order correlation

Verification of monotonic convergence prior to extrapolation is strongly advised.


Limitations
-----------

CBS extrapolation assumes smooth and monotonic convergence with increasing
basis size.

It may become unreliable for:

- Strong multireference systems
- Inconsistent basis families
- Very small basis sets
- Explicitly correlated (F12) methods without adapted modeling

