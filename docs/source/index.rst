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
--------
* **Extrapolation:** Support for multiple CBS extrapolation schemes.
* Hartree-Fock extrapolations schemes: Jensen, Klopper, Feller, Truhlar, HF-E
* Correlation Energy extrapolations schemes: Truhlar, Huh-Lee, Bakowies, OANc, Halkier-Helgaker, Martin, USPE, USTE
* **Optimization:** High-dimensional optimization using L-BFGS-B and SQM.
* **Flexibility:** input file based configuration for reproducible workflows.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Configuration Reference

   configuration

.. toctree::
   :maxdepth: 1
   :caption: Academic Record

   citing

