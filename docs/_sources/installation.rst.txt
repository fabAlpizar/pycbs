Installation
============

.. _installation:

This project is designed to work within a **Conda-based environment**.
All dependencies are managed through the provided ``environment.yml`` file.

Prerequisites
-------------

Ensure you have one of the following installed:

- Anaconda
- Miniconda
- Mambaforge

Verify your installation:

.. code-block:: bash

   conda --version

From Source
-----------

If you want to contribute, modify the code, or use the latest development version,
install directly from the GitHub repository.

Clone the repository

.. code-block:: bash

   git clone https://github.com/fabAlpizar/pycbs.git

Navigate into the project directory

.. code-block:: bash

   cd pycbs

Create the Conda environment

This installs all required dependencies in an isolated environment.

.. code-block:: bash

   conda env create -f environment.yml

Activate the environment

.. code-block:: bash

   conda activate pycbs

Install in editable mode

Editable installation allows live code modifications without reinstalling.

.. code-block:: bash

   pip install -e .

Notes
-----

- Enjoy.