---
title: ILAB TEMPLATE - Data Science
purpose: Template for python projects tailored to scientific applications (e.g., machine learning)
---

## Astrotime

#### Machine learning methods for irregularly spaced time series

### Conda environment

* On Adapt load modules: gcc/12.1.0, nvidia/12.1
* If mamba is not available, install [miniforge](https://github.com/conda-forge/miniforge) (or load mamba module)
* Execute the following to set up a conda environment for astrotime:

    >   * mamba create -n astrotime ninja python=3.10
    >   * mamba activate astrotime
    >   * pip install tensorflow[and-cuda] tensorboard jupyterlab==4.0.13 ipywidgets==7.8.4 jupyterlab_widgets ipykernel==6.29 ipympl ipython==8.26 xarray netCDF4
    >   * pip install hydra-core rich  scikit-learn

