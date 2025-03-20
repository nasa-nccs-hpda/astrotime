# Astrotime

#### Machine learning methods for irregularly spaced time series

## Project Description

This project contains the implementation of a set of time-aware neural network (TAN) and workflows for testing their performance on the task of predicting periods of the sinusoidal timeseries dataset (STD) provided by Brian Powell.   Its performance on this dataset (and a 50% reduced version) was compared with the performance of the baseline CNN network (BCN) provided by Brian Powell.
The BCN operates directly on the timeseries values (without the time information).   The TAN utilizes the same network as the BCN but operates on a weighted projection of the timeseries onto a set of sinusoidal basis functions, which enfolds both value and time components.
When tested on the unmodified STD, the BCN achieved a mean absolute error (MAE) of 0.03 and the TAN achieved a MAE of 0.01.   Because the STD is close to being regularly sampled, the BCN (which implicitly assumes regularly sampled data) performs reasonably well, and the addition of time information in the TAN yields a relatively small improvement.
To compare the performance of these models on a (more) irregularly sampled dataset, we subsampled the STD by randomly removing 50% of the observations.   On the sparse STD the TAN again achieved a MAE of 0.02, but the BCN performance was greatly degraded, resulting in a MAE of 0.25.   These results verify that the TAN is effectively using the time information of the dataset, whereas the BCN is operating on the shape of the value curve assuming regularly sampled observations.

### Analysis vs. Synthesis

* This project implements two forms of wavelet transform: an analysis transform and a synthesis transform.
* The analysis coefficients represent the projection of a signal onto a set of basis functions, implemented as a weighted inner product between the signal and the basis functions (evaluated at the time points).  
* The synthesis coefficients represent the optimal representation of the signal as a weighted sum of the basis functions (i.e. the minimum error projection).
* If the basis functions are orthogonal, then the analysis and synthesis coefficients are the same (as in the FFT). However, when the time points are irregular, then the basis functions (evaluated at the time points) are never orthogonal, and additional computation is required to generate the synthesis coefficients.

### Model Equations

There is a good summary of the equations implemented in this project in the appendix of [Witt & Schumann (2005)](https://www.researchgate.net/publication/200033740_Holocene_climate_variability_on_millennial_scales_recorded_in_Greenland_ice_cores).   
The wavelet synthesis transform generates two features described by equations A10 and A11.  
The wavelet analysis transform generates three features by computing weighted scalar products (equation A3) between the signal values and the sinusoid basis functions described by equation A5.  
Equation A7 shows the relationship between the analysis and synthesis coefficients.
Futher mathematical detail can be found in [Foster (1996)](https://articles.adsabs.harvard.edu/pdf/1996AJ....112.1709F).

## Conda environment

* On Adapt load modules: gcc/12.1.0, nvidia/12.1
* If mamba is not available, install [miniforge](https://github.com/conda-forge/miniforge) (or load mamba module)
* Execute the following to set up a conda environment for astrotime:

### Torch Environment 

    >   * mamba create -n astrotime.pt ninja python=3.10
    >   * mamba activate astrotime
    >   * pip install torch jupyterlab==4.0.13 ipywidgets==7.8.4 cuda-python jupyterlab_widgets ipykernel==6.29 ipympl ipython==8.26 xarray netCDF4
    >   * pip install torchcde torchdiffeq hydra-core rich  scikit-learn

## Dataset Preparation

* The project data directory on explore is: **/explore/nobackup/projects/ilab/data/astrotime**.
* This project uses a baseline dataset of artificially generated sinusoids, downloadable from a [sharepoint folder](https://nasa-my.sharepoint.com/:f:/r/personal/bppowel1_ndc_nasa_gov/Documents/sinusoids?e=5%3af465681647e04bdca4ed910aec775237&sharingv2=true&fromShare=true&xsdata=MDV8MDJ8dGhvbWFzLm1heHdlbGxAbmFzYS5nb3Z8NTg3YmJmMjkyMjhlNDliNTQ4MTEwOGRkMWY5YTJjNzZ8NzAwNWQ0NTg0NWJlNDhhZTgxNDBkNDNkYTk2ZGQxN2J8MHwwfDYzODcwMTQ2OTIwMTE0MzU4NXxVbmtub3dufFRXRnBiR1pzYjNkOGV5SkZiWEIwZVUxaGNHa2lPblJ5ZFdVc0lsWWlPaUl3TGpBdU1EQXdNQ0lzSWxBaU9pSlhhVzR6TWlJc0lrRk9Jam9pVFdGcGJDSXNJbGRVSWpveWZRPT18MHx8fA%3d%3d&sdata=YXprclJBZFpZcmhRMlRoYUJQbWdDeEpjMldBSEh6MTlFNllsYkNDWDAvVT0%3d).
* The raw dataset has been downloaded to explore at: **{datadir}/sinusoids/npz**.
* The script **.workflow/npz2nc.py** has been used to convert the .npz files to netcdf format.
* The netcdf files, which are used in this project's ML workflows, can be found at: **{datadir}/sinusoids/nc**.

## Workflows
This project provides three ML workflows:

*   _Baseline_ (**workflow/train-baseline-cnn.py**):  This workflow runs the baseline CNN (developed by Brian Powell) which takes only timeseries value data as input.
*   _Wavelet Synthesis_ (**workflow/wavelet-synthesis-cnn.py**): This workflow runs the same baseline CNN operating on a weighted wavelet z-transform, which enfolds both the time and value data from the timeseries. 
*   _Wavelet Analysis_ (**workflow/wavelet-analysis-cnn.py**): This workflow runs the same baseline CNN operating a projection of the timeseries onto a set of sinusoid basis functions, which enfolds both the time and value data from the timeseries. 

The *_small versions execute the workflows on a subset (1/10) of the full training dataset.
The workflows save checkpoint files at the end of each epoch.  By default the model is initialized with any existing checkpoint file at the begining of script execution.  To
execute the script with a new set of checkpoints (while keeping the old ones), create a new script with a different value of the *version* parameter 
(and a new defaults hydra yaml file with the same name in the config dir).  

## Configuration

The workflows are configured using [hydra](https://hydra.cc/docs/intro/).
* All hydra yaml configuration files are found under **config**.
* The workflow configurations can be modified at runtime as [supported by hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).
* For example, the following command runs the baseline workflow on gpu 3 with random initialization (i.e. ignoring & overwriting any existing checkpoints):
    >   python workflow/baseline-cnn.py platform.gpu=3 train.refresh_state=True
* To run validation (no training), execute:
    >   python workflow/baseline-cnn.py train.mode=valid platform.gpu=0

### Configuration Parameters

Here is a partial list of configuration parameters with typical default values.  Their values are configured in the hydra yaml files and reconfigurable on the command line:

       platform.project_root:  "/explore/nobackup/projects/ilab/data/astrotime"   # Base directory for all saved files
       platform.gpu: 0                                                            # Index of gpu to execcute on
       platform.log_level: "info"                                                 # Log level: typically debug or info
       data.source: sinusoid                                            # Dataset type (currently only sinusoid is supported)
       data.dataset_root:  "${platform.project_root}/sinusoids/nc"      # Location of processed netcdf files
       data.dataset_files:  "padded_sinusoids_*.nc"                     # Glob pattern for file names
       data.file_size: 1000                                             # Number of sinusoids in a single nc file
       data.batch_size: 50                                              # Batch size for training
       data.validation_fraction: 0.1                                    # Fraction of training dataset that is used for validation
       data.dset_reduction: 1.0                                         # Fraction of the full dataset that is used for training/validation
       transform.series_length: 1536                         # Length of subset of input timeseries to process
       transform.nfeatures: 1                                # Number of feaatures to be passed to network
       transform.sparsity: 0.0                               # Fraction of observations to drop (randomly)
       model.cnn_channels: 64                                # Number of channels in first CNN layer
       model.dense_channels: 64                              # Number of channels in dense layer
       model.out_channels: 1                                 # Number of network output channels
       model.num_cnn_layers: 3                               # Number of CNN layers in a CNN block
       model.num_blocks: 7                                   # Number of CNN blocks in the network
       model.pool_size: 2                                    # Max pool size for every block
       model.stride: 1                                       # Stride value for every CNN layer
       model.kernel_size: 3                                  # Kernel size for every CNN layer
       model.cnn_expansion_factor: 4                         # Increase in the number of channels from one CNN layer to the next
       train.optim: rms                                              # Optimizer
       train.lr: 1e-3                                                # Learning rate
       train.nepochs: 5000                                           #  Training Epochs
       train.refresh_state: False                                    # Start from random weights (Ignore & overwrite existing checkpoints)
       train.overwrite_log: True                                     # Start new log file
       train.results_path: "${platform.project_root}/results"        # Checkpoint and log files are saved under this directory
       train.weight_decay: 0.0                                       # Weight decay parameter for optimizer
       train.mode:  train                                            # execution mode: 'train' or 'valid'

## Working from the container

In addition to the anaconda environment, the software can be run from
a container. This project provides a Docker container that can be converted
to Singularity or any container engine based on the user needs. The 
instructions below are geared towards the use of Singularity since that is 
the default available in the NCCS super computing facility.

### Container Download

To create a sandbox out of the container:

```bash
singularity build --sandbox /lscratch/$USER/container/astrotime docker://nasanccs/astrotime:latest
```

An already downloaded version of this sandbox is available under:

```bash
TBD: Need to move the container to the ILAB container space
```

### Working from the container with a shell session

To get a shell session inside the container:

```bash
singularity shell -B $NOBACKUP,/explore/nobackup/projects,/explore/nobackup/people --nv /lscratch/jacaraba/container/astrotime
```

An example run training:

```bash
PYTHONPATH="/explore/nobackup/people/jacaraba/development/astrotime" python /explore/nobackup/people/jacaraba/development/astrotime/work
flow/baseline-cnn.py platform.project_root="/explore/nobackup/projects/ilab/scratch/jacaraba/astrotime" data.dataset_root="/explore/nobackup/projects
/ilab/data/astrotime/sinusoids/nc"
```

An example run validation:

```bash
PYTHONPATH="/explore/nobackup/people/jacaraba/development/astrotime" python /explore/nobackup/people/jacaraba/development/astrotime/work
flow/baseline-cnn.py platform.project_root="/explore/nobackup/projects/ilab/scratch/jacaraba/astrotime" data.dataset_root="/explore/nobackup/projects
/ilab/data/astrotime/sinusoids/nc" train.mode=valid
```

PYTHONPATH="/explore/nobackup/people/jacaraba/development/astrotime" python /explore/nobackup/people/jacaraba/development/astrotime/workflow/baseline-cnn.py platform.project_root="/explore/nobackup/projects/ilab/scratch/jacaraba/astrotime" data.dataset_root="/explore/nobackup/projects/ilab/data/astrotime/sinusoids/nc"

### Sending a slurm job using the container

```bash
TBD
```

## References

- Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
- Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores. Nonlinear Processes in Geophysics 12, 345–352 (2005).
