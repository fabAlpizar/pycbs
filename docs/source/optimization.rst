Geometrical optimization
------------------------

Gradient based
~~~~~~~~~~~~~~
.. code-block:: ini

    [optimization3]
    optimization = True
    input_xyz = path/to/xyz/file
    method = CCSD(T)
    optimizer = L-BFGS-B
    basis1 = cc-pvdz
    basis2 = cc-pvtz
    spin = 0
    x1 = 1.852
    x2 = 2.639
    x1hf = 3.02
    x2hf = 3.64

SQM
~~~
.. code-block:: ini

    [optimization3]
    optimization = True
    input_xyz = path/to/xyz/file
    method = CCSD(T)
    optimizer = sqm
    basis1 = cc-pvdz
    basis2 = cc-pvtz
    spin = 0
    x1 = 1.852
    x2 = 2.639
    x1hf = 3.02
    x2hf = 3.64
