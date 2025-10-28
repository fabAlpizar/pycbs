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


1. Create a conda environment for pyCBS:
   ```
   conda create -n pyCBS python=3.10
   ```
>**_Note:_** You can change the Python version if needed
2. Activate the new environment with:
   ```
   conda activate pyCBS
   ```
3. Clone this repo into the environment:
   ```
   git clone https://github.com/fabAlpizar/pyCBS
   cd pyCBS
   ```
4. Review and uncomment what you need in `environment.yml` and create an environment `pyCBS` with the help of [conda]:
   ```
   conda env update -f environment.yml
   ```
   _(or if you didn't create the environment yet:)_
   ```
   conda env create -f environment.yml

   ```
   


> **_NOTE:_**  The conda environment will have pyCBS installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.





# 📬 Contact

For questions, feature requests, or bug reports, please open an issue at:
👉 https://github.com/fabAlpizar/pyCBS/issues




## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── pycbs               <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

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
