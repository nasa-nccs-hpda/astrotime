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

###### Torch Environment

    >   * mamba create -n astrotime.pt ninja python=3.10
    >   * mamba activate astrotime
    >   * pip install torch jupyterlab==4.0.13 ipywidgets==7.8.4 cuda-python jupyterlab_widgets ipykernel==6.29 ipympl ipython==8.26 xarray netCDF4
    >   * pip install hydra-core rich  scikit-learn

###### Tensorflow Environment (Deprecated)

    >   * mamba create -n astrotime.tf ninja python=3.10
    >   * mamba activate astrotime
    >   * pip install tensorflow[and-cuda] tensorboard jupyterlab==4.0.13 ipywidgets==7.8.4 jupyterlab_widgets ipykernel==6.29 ipympl ipython==8.26 xarray netCDF4
    >   * pip install hydra-core rich termcolor scikit-learn

### Dataset Preparation

* The project data directory on explore is: **/explore/nobackup/projects/ilab/data/astrotime**.
* This project uses a baseline dataset of artificially generated sinusoids, downloadable from a [sharepoint folder](https://nasa-my.sharepoint.com/:f:/r/personal/bppowel1_ndc_nasa_gov/Documents/sinusoids?e=5%3af465681647e04bdca4ed910aec775237&sharingv2=true&fromShare=true&xsdata=MDV8MDJ8dGhvbWFzLm1heHdlbGxAbmFzYS5nb3Z8NTg3YmJmMjkyMjhlNDliNTQ4MTEwOGRkMWY5YTJjNzZ8NzAwNWQ0NTg0NWJlNDhhZTgxNDBkNDNkYTk2ZGQxN2J8MHwwfDYzODcwMTQ2OTIwMTE0MzU4NXxVbmtub3dufFRXRnBiR1pzYjNkOGV5SkZiWEIwZVUxaGNHa2lPblJ5ZFdVc0lsWWlPaUl3TGpBdU1EQXdNQ0lzSWxBaU9pSlhhVzR6TWlJc0lrRk9Jam9pVFdGcGJDSXNJbGRVSWpveWZRPT18MHx8fA%3d%3d&sdata=YXprclJBZFpZcmhRMlRoYUJQbWdDeEpjMldBSEh6MTlFNllsYkNDWDAvVT0%3d).
* The raw dataset has been downloaded to explore at: **{datadir}/sinusoids/npz**.
* The script **.workflow/npz2nc.py** has been used to convert the .npz files to netcdf format.
* The netcdf files, which are used in this project's ML workflows, can be found at: **{datadir}/sinusoids/nc**.

### Workflows
This project provides two ML workflows:
* _Baseline_ (**.workflow/train-baseline-cnn.py**):  This workflow runs the baseline CNN (developed by Brian Powell) which takes only timeseries value data as input.
* _WWZ_ (**.workflow/train-wwz-cnn.py**): This workflow runs the same baseline CNN operating on a weighted wavelet z-transform, which enfolds both the time and value data from the timeseries.

### Configuration

The workflows are configured using [hydra](https://hydra.cc/docs/intro/).
* All configuration files are found under **.config**.
* The workflow configurations can be modified at runtime as [supported by hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).
* For example, the following command runs the baseline workflow on gpu 3 with random initialization (i.e. ignoring any existing checkpoints):
    >   python .workflow/train-baseline-cnn.py platform.gpu=3 train.refresh_state=True