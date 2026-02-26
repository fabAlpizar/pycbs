.. pyCBS documentation master file, created by
   sphinx-quickstart on Mon Feb  9 13:48:54 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyCBS: Python Complete Basis Set Tools
=====================================

.. image:: _static/logo_pycbs.jpeg
   :align: center
   :width: 200px

**pyCBS** is a high-performance Python package designed to perform Complete Basis Set extrapolations accros several schemes. 
It also provides a novel geometrical optimization implementation via de Succesive Quadratic Minimization (SQM) or the gradient based L-BFGS-B method, in which you obtain both CBS-optimized geometry and energy .

Features
========

.. _features:

**1. Extrapolation**

Support for multiple Complete Basis Set (CBS) extrapolation schemes,
organized by energy component.

1.1 Hartree–Fock schemes

- Jensen
- Klopper
- Feller
- Truhlar
- HF-E

1.2 Correlation energy schemes

- Truhlar
- Huh–Lee
- Bakowies
- OANc
- Halkier–Helgaker
- Martin
- USPE
- USTE


**2. Optimization**

- L-BFGS-B
- SQM


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
.. toctree::
   :maxdepth: 2
   :caption: Theory & Computation

   TC
   basis
   hierarchical


.. toctree::
   :maxdepth: 2
   :caption: Setting up the Config File

   extrapolations
   optimization

.. toctree::
   :maxdepth: 2
   :caption: Analyze the Results

   outputs
   opto


.. toctree::
   :maxdepth: 1
   :caption: Academic Record

   citing

