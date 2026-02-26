Output Analysis Report
======================

.. _output_analysis_report:

Summary
-------

Provide a concise summary of the computational results.

Key Metrics
-----------

+----------------------+----------------+
| Quantity             | Value          |
+======================+================+
| HF Energy (CBS)      |                |
+----------------------+----------------+
| Correlation Energy   |                |
+----------------------+----------------+
| Total Energy         |                |
+----------------------+----------------+

Energy Convergence Analysis
----------------------------

Convergence Behavior
~~~~~~~~~~~~~~~~~~~~

Describe how the energy changes with basis set size.

.. figure:: images/energy_convergence.png
   :width: 80%
   :align: center
   :alt: Energy convergence trend

   Total energy convergence with increasing basis set cardinal number.

Error Analysis
--------------

Absolute Error
~~~~~~~~~~~~~~

.. math::

   \Delta E = |E_n - E_{\text{CBS}}|

.. figure:: images/error_decay.png
   :width: 70%
   :align: center
   :alt: Error decay curve

   Error decay relative to the extrapolated CBS limit.

Residual Analysis
~~~~~~~~~~~~~~~~~

Insert diagnostics for fitting quality.

.. figure:: images/residuals.png
   :width: 70%
   :align: center
   :alt: Fit residuals

   Residual distribution of extrapolation model.

Statistical Indicators
----------------------

- R²:
- RMSE:
- Fit exponent:
- Confidence interval:

Reproducibility
---------------

- Software version:
- Basis set family:
- Hardware specifications:
- Date of calculation:

Conclusions
-----------

Summarize:

- Accuracy achieved
- Convergence reliability
- Recommended production setup

Appendix
--------

Additional plots can be inserted here:

.. figure:: images/additional_plot.png
   :width: 75%
   :align: center

   Optional extended diagnostics.