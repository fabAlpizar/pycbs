.. pyCBS documentation master file, created by
   sphinx-quickstart on Mon Feb  9 13:48:54 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyCBS: Python Complete Basis Set Tools
=====================================

.. image:: _static/pyCBS.png
   :align: center
   :width: 200px

**pyCBS** is a high-performance Python package designed to perform Complete Basis Set (CBS) extrapolations accros several schemes. 


Features
========

.. _features:

**1. Extrapolation**

The software provides full support for multiple CBS extrapolation schemes, which are systematically organized by energy component. To achieve high-precision energy limits, the calculations are divided into two primary contributions:

- Hartree-Fock (HF) Energy Schemes: Includes specialized formulations for the mean-field energy convergence, featuring methods by Jensen, Klopper, Feller, Truhlar and Varandas.

- Correlation Energy Schemes: Dedicated to capturing electron correlation effects through advanced models such as Truhlar, Huh-Lee, Bakowies, Okoshi-Atsumi-Nakai, Halkier-Helgaker, Martin and Varandas.



**2. Optimization**

The program features robust geometry and structural optimization capabilities. Users can utilize advanced mathematical minimization and electronic structure refinement schemes, including:

- L-BFGS-B gradient-based optimization
- Successive Quadratic Minimization (SQM)


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
   :maxdepth: 1
   :caption: Academic Record

   citing

