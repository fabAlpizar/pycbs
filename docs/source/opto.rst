Output Files Overview
=====================

.. _output_files_overview:

Introduction
------------

This section describes the generated output files, their structure,
data content, and visual representations.

Directory Structure
--------------------

.. code-block:: text

   outputs/
   ├── energies.dat
   ├── frequencies.csv
   ├── extrapolation_summary.txt
   ├── plots/
   │   ├── hf_extrapolation.png
   │   └── correlation_extrapolation.png

File Descriptions
-----------------

energies.dat
~~~~~~~~~~~~

- Contains total electronic energies.
- Units:
- Format:
- Example entry:

.. code-block:: text

   # Method   Basis    Energy (Ha)
   HF        cc-pVTZ  -76.345678

frequencies.csv
~~~~~~~~~~~~~~~

- Vibrational frequencies
- Columns:
- Units:

Visualization
-------------

HF Extrapolation Plot
~~~~~~~~~~~~~~~~~~~~~

.. figure:: images/hf_extrapolation.png
   :width: 75%
   :align: center
   :alt: HF extrapolation curve

   Hartree–Fock energy extrapolation as a function of cardinal number.

Correlation Energy Plot
~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: images/correlation_extrapolation.png
   :width: 75%
   :align: center
   :alt: Correlation extrapolation curve

   Correlation energy convergence behavior.

Interpretation Guidelines
-------------------------

- Verify monotonic convergence.
- Check extrapolated limit consistency.
- Compare against reference benchmarks.

Common Issues
-------------

- Non-monotonic energy behavior
- SCF convergence failures
- Numerical precision artifacts

Cross-References
----------------

See also:

- :ref:`theoretical_background`
- :ref:`basis_sets`