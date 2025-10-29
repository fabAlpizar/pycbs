[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/pyCBS.svg?branch=main)](https://cirrus-ci.com/github/<USER>/pyCBS)
[![ReadTheDocs](https://readthedocs.org/projects/pyCBS/badge/?version=latest)](https://pyCBS.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/pyCBS/main.svg)](https://coveralls.io/r/<USER>/pyCBS)
[![PyPI-Server](https://img.shields.io/pypi/v/pyCBS.svg)](https://pypi.org/project/pyCBS/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/pyCBS.svg)](https://anaconda.org/conda-forge/pyCBS)
[![Monthly Downloads](https://pepy.tech/badge/pyCBS/month)](https://pepy.tech/project/pyCBS)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/pyCBS)
-->


<p align="">
  <img src="images/img.png" width="110" alt="pycbs logo">
</p>

# pyCBS

> pyCBS: A tool to perform Complete Basis Set calculations 

# Documentation 

# Official Paper

## Features

- Perform CBS extrapolations for energy and geometry.
- Supports HF, MP2, and CCSD(T) methods.
- Automates multi-basis calculations.
- Saves all intermediate outputs in an organized directory structure.
- Compatible with PySCF for reliable quantum chemistry calculations.
- Easy to extend with additional basis sets or methods.



## Installation

In order to set up the necessary environment:


1. Clone this repository:
   ```
   git clone https://github.com/fabAlpizar/pyCBS
   cd pyCBS
   ```
2. Create the new environment with:
   ```
   conda env create -f environment.yml
   ```
3. Activate the new environment with: 
   ```
   conda activate pyCBS
   ```
4. (Optional) If you modify code or configuration files and want changes to take effect, reinstall in editable mode:
   ```
   pip install -e.
   ```
   
   


> **_NOTE:_**  The conda environment will have pyCBS installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.





# Contact

For questions, feature requests, or bug reports, please open an issue at:
👉 https://github.com/fabAlpizar/pyCBS/issues






<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.6 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
