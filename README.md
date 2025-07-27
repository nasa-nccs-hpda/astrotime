# Astrotime

#### Machine learning methods for irregularly spaced time series

## Quick Start

For a quick start, workflows and container usage have been documented in this section. For additional
details, please read the rest of the sections of this README. As a summary, each workflow 
(Sinusoid, Synthetic, and MIT) have a training, eval, and peakfinder script. 

For the MIT dataset, the train step is replaced with finetune, because in this case the training is
intended to start with weights from the synthetic training. The peakfinder scripts run a simple (non-ML)
workflow that computes the frequency of the highest peak in the spectrum, and returns the corresponding 
period, which is used for comparison and evaluation of the ML workflow.

### Sinusoid Dataset Workflow

An example run training the deep learning model:

```bash
PYTHONPATH=/explore/nobackup/people/jacaraba/development/astrotime python /explore/nobackup/people/jacaraba/development/astrotime/workflow/release/sinusoid/train.py platform.project_root=/explore/nobackup/projects/ilab/ilab_testing/jacaraba/astrotime data.dataset_root=/explore/nobackup/projects/ilab/data/astrotime/sinusoids/nc train.nepochs=10 data.batch_size=16
```

Note that the following are the options allowed to run this workflow. If you need to change the path to the data or any other settings,
feel free to modify the settings coming from the CLI.

```bash
Singularity> python /explore/nobackup/people/jacaraba/development/astrotime/workflow/full/sinusoid/train.py -h
train is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

__legacy__: MIT_period, MIT_period.ce, MIT_period.octaves, MIT_period.octaves.pcross, MIT_period.synthetic, MIT_period.synthetic.folded, MIT_period.wp, baseline_cnn, desktop_period.analysis, desktop_period.octaves, progressive_MIT_period, sinusoid_period.baseline, sinusoid_period.baseline_small, sinusoid_period.poly, sinusoid_period.wp, sinusoid_period.wp_scaled, sinusoid_period.wp_small, sinusoid_period.wpk, sinusoid_period.wwz, sinusoid_period.wwz_small, synthetic_period_autocorr, synthetic_period_transformer, synthetic_period_transformer.classification, synthetic_period_transformer.regression, synthetic_transformer
__legacy__/data: MIT, MIT-1, MIT.csv, MIT.octaves, MIT.synthetic, MIT.synthetic.folded, astro_synthetic, astro_synthetic_autocorr, pcross.octaves, planet_crossing_generator, sinusoids.nc, sinusoids.npz, sinusoids_small.nc
__legacy__/model: relation_aware_transformer, transformer, transformer.classication, transformer.regression, wpk_cnn
__legacy__/transform: MIT.octaves, MIT.synthetic, MIT.synthetic.folded, ce-MIT, correlation, gp, value, wp, wp-MIT, wp-scaled, wpk, wwz
data: MIT, sinusoids, synthetic, synthetic.octave
model: cnn, cnn.classification, cnn.octave_regression, dense
platform: desktop1, explore
train: MIT_cnn, sinusoid_cnn, synthetic_cnn
transform: MIT, sinusoid, synthetic, synthetic.octave


== Config ==
Override anything in the config (foo.bar=value)

platform:
  project_root: /explore/nobackup/projects/ilab/data/astrotime
  gpu: 0
  log_level: info
train:
  optim: rms
  lr: 0.001
  nepochs: 5000
  refresh_state: false
  overwrite_log: true
  results_path: ${platform.project_root}/results
  weight_decay: 0.0
  mode: train
  base_freq: ${data.base_freq}
transform:
  sparsity: 0.0
  batch_size: ${data.batch_size}
  nfreq_oct: ${data.nfreq_oct}
  base_freq: ${data.base_freq}
  noctaves: ${data.noctaves}
  test_mode: ${data.test_mode}
  maxh: ${data.maxh}
  accumh: false
  decay_factor: 0.0
  subbatch_size: 4
  norm: std
  fold_octaves: false
data:
  source: sinusoid
  dataset_root: ${platform.project_root}/sinusoids/nc
  dataset_files: padded_sinusoids_*.nc
  cache_path: ${platform.project_root}/cache/data/synthetic
  dset_reduction: 1.0
  batch_size: 16
  nfreq_oct: 512
  base_freq: 0.025
  noctaves: 9
  test_mode: default
  file_size: 1000
  nfiles: 1000
  refresh: false
  maxh: 8
model:
  mtype: cnn.regression
  cnn_channels: 64
  dense_channels: 64
  out_channels: 1
  num_cnn_layers: 3
  num_blocks: 8
  pool_size: 2
  stride: 1
  kernel_size: 3
  cnn_expansion_factor: 4
  base_freq: ${data.base_freq}
  feature: 1


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```

Then followed by the peakfinder method:

```bash
PYTHONPATH=/explore/nobackup/people/jacaraba/development/astrotime python /explore/nobackup/people/jacaraba/development/astrotime/workflow/release/sinusoid/peakfinder.py platform.project_root=/explore/nobackup/projects/ilab/ilab_testing/jacaraba/astrotime data.dataset_root=/explore/nobackup/projects/ilab/data/astrotime/sinusoids/nc
```

Finally, performing evaluation of these methods:

```bash
PYTHONPATH=/explore/nobackup/people/jacaraba/development/astrotime python /explore/nobackup/people/jacaraba/development/astrotime/workflow/release/sinusoid/eval.py platform.project_root=/explore/nobackup/projects/ilab/ilab_testing/jacaraba/astrotime data.dataset_root=/explore/nobackup/projects/ilab/data/astrotime/sinusoids/nc train.nepochs=10 data.batch_size=16
```

### Synthetic Dataset Workflow

```bash
PYTHONPATH=/explore/nobackup/people/jacaraba/development/astrotime python /explore/nobackup/people/jacaraba/development/astrotime/workflow/release/synthetic/train.py platform.project_root=/explore/nobackup/projects/ilab/ilab_testing/jacaraba/astrotime data.dataset_root=/explore/nobackup/projects/ilab/data/astrotime/sinusoids/nc train.nepochs=10 data.batch_size=16
```

### MIT Dataset Workflow

```bash
```

## Project Description

This project contains the implementation of a time-aware neural network (TAN) and workflows for testing its performance on the task of predicting periods of the timeseries datasets provided by Brian Powell.  
Three datasets have been provided by Brian Powell for test and evalutaion:
  * Synthetic Sinusoids (SS):     A set of sinusoid timeseries with irregular time spacing. 
  * Synthetic Light Curves (SLC): A set of artifically generated timeseries imitating realistic lightcurves. 
  * MIT Lightcurves (MIT-LC):     A set of actual lightcurves provided by MIT.

### Spectral Projection

* This project utilizes a spectral projection as the first stage of data processing. The spectral coefficients represent the projection of a signal onto a set of basis functions, 
  implemented as a weighted inner product between the signal and the basis functions (evaluated at the time points). There is a good summary of the equations implemented in this project 
  in the appendix of [Witt & Schumann (2005)](https://www.researchgate.net/publication/200033740_Holocene_climate_variability_on_millennial_scales_recorded_in_Greenland_ice_cores). 
  The spectral projection generates three features by computing weighted scalar products (equation A3) between the signal values and the sinusoid basis functions described by equation A5.  
  The magnitude of the projection is defined by equation A10.  Futher mathematical detail can be found in [Foster (1996)](https://articles.adsabs.harvard.edu/pdf/1996AJ....112.1709F).
* The frequency (f) space is scaled such that the density of f valuse is constant across octaves.  
  The f values are given by f[j] = f0 * pow( 2, j/N ), with j ranging over [0,N*M], where N is the number of f values per octave, 
  M is the number of octaves in the f range, and f0 is the lowest value in the f range. 

### Learning Model
* This project utilizes a convolutional neural network (CNN) with 24 layers.  For each of the datasets, the input to the network is the spectral projection of each light curve (LC) 
   and the output is the frequency of a periodic component of the LC, trained using the target frequency provided in the dataset for each LC.  
* The output layer of the network is dense, with an exponential activation function defined by the equation y = f0 * (pow(2, x) - 1), where f0 is the lowest value in the f range. 
  In order to account for the very large dynamic range of the target frequency spectrum, a custom loss function is used, defined by the equation 
  loss = abs( log2( (yn + f0) / (yt + f0) ) ), where yn is the network output and yt is the target frequency.

## Conda environment

* On Adapt load modules: gcc/12.1.0, nvidia/12.1
* If mamba is not available, install [miniforge](https://github.com/conda-forge/miniforge) (or load mamba module)
* Execute the following to set up a conda environment for astrotime:

### Torch Environment:

    >   * mamba create -n astrotime.pt ninja python=3.10
    >   * mamba activate astrotime
    >   * pip install torch jupyterlab==4.0.13 ipywidgets==7.8.4 cuda-python jupyterlab_widgets ipykernel==6.29 ipympl ipython==8.26 xarray netCDF4 pygam wotan statsmodels transitleastsquares scikit-learn hydra-core rich 
    >   * pip install diffusers lightkurve --upgrade

## Dataset Preparation

* This project utilizes three datasets (sinusoid, synthetic, and MIT) which are located in the **cfg.platform.project_root** directory. The project_root directory on explore is: **/explore/nobackup/projects/ilab/data/astrotime**.
* The raw sinusoid data can be found on explore at <project_root>/sinusoids/npz.  The script **.workflow/util/npz2nc.py** has been used to convert the .npz files to netcdf files in the  <project_root>/sinusoids/nc directory.
* The raw synthetic light curves are stored on explore at **/explore/nobackup/people/bppowel1/timehascome/**. The script **.workflow/util/npz2nc.py** has been used to convert the .npz files to netcdf files in the <project_root>/synthetic directory.
* The MIT light curves are stored in their original form at: **/explore/nobackup/people/bppowel1/mit_lcs/**. Methods in the class **astrotime.loaders.MIT.MITLoader** have been used to convert the lc txt files to netcdf files in the <project_root>/MIT directory.

## Workflows
For each of the datasets (sinusoid, synthetic, and MIT), three ML workflows are provided:

*   _train_ (**.workflow/train-baseline-cnn.py**):        Runs the TAN training workflow.
*   _eval_ (**.workflow/wavelet-synthesis-cnn.py**):      Runs the TAN validation/test workflow.
*   _peakfinder_ (**.workflow/wavelet-analysis-cnn.py**): Runs the peakfinder validation/test workflow.

The workflows save checkpoint files at the end of each epoch.  By default the model is initialized with any existing checkpoint file at the begining of script execution. 
A workflow's checkpoints are named after it's *version* parameter.
To execute the script with a new set of checkpoints (while keeping the old ones), create a new script with a different value of the *version* parameter 
(and a new defaults hydra yaml file with the same name in the config dir).   The second (ckp_version) argument to the _train_ method of the Trainer class is used for fine
tuning.  If this argument is specified, then the training workflow will be initialized with the checkpoint from that version, and all new checkpoint saves will be
to the primary version of the workflow.

## Configuration

The workflows are configured using [hydra](https://hydra.cc/docs/intro/).
* All hydra yaml configuration files are found under **.config**.
* The workflow configurations can be modified at runtime as [supported by hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).
* For example, the following command runs the synthetic dataset training workflow on gpu 3 with random initialization (i.e. ignoring & overwriting any existing checkpoints):
    >   python workflow/synthetic/train.py platform.gpu=3 train.refresh_state=True
* To run validation (no training), execute:
    >   python workflow/synthetic/train.py train.mode=valid platform.gpu=0

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
*note - /lscratch is only available on gpu### nodes

An already downloaded version of this sandbox is available under:

```bash
/explore/nobackup/projects/ilab/containers/astrotime-latest
```

### Working from the container with a shell session

To get a shell session inside the container:

```bash
singularity shell -B $NOBACKUP,/explore/nobackup/projects,/explore/nobackup/people --nv /explore/nobackup/projects/ilab/containers/astrotime-latest
```

### An example run training

An example run training:

```bash
python /explore/nobackup/projects/ilab/ilab_testing/astrotime/workflow/baseline-cnn.py platform.project_root=/explore/nobackup/projects/ilab/ilab_testing/astrotime data.dataset_root=/explore/nobackup/projects/ilab/data/astrotime/sinusoids/nc
```
Expected training output files:
```bash
/explore/nobackup/projects/ilab/ilab_testing/astrotime/results/checkpoints/sinusoid_period.baseline.pt
/explore/nobackup/projects/ilab/ilab_testing/astrotime/results/checkpoints/sinusoid_period.baseline.backup.pt
```

An example run validation:

```bash
python /explore/nobackup/projects/ilab/ilab_testing/astrotime/workflow/baseline-cnn.py platform.project_root=/explore/nobackup/projects/ilab/ilab_testing/astrotime data.dataset_root=/explore/nobackup/projects/ilab/data/astrotime/sinusoids/nc train.mode=valid
```
Expected validation output:
```bash
      Loading checkpoint from /explore/nobackup/projects/ilab/ilab_testing/astrotime/results/checkpoints/sinusoid_period.baseline.pt: epoch=122, batch=0

SignalTrainer[TSet.Validation]: 2000 batches, 1 epochs, nelements = 100000, device=cuda:0
 Validation Loss: mean=0.021, median=0.021, range=(0.012 -> 0.043)
98.04user 8.85system 2:00.79elapsed 88%CPU (0avgtext+0avgdata 1080416maxresident)k
2059752inputs+1120outputs (1677major+582379minor)pagefaults 0swaps
```

### Sending a slurm job using the container (training example):

From gpulogin1:

```bash
sbatch --mem-per-cpu=10240 -G1 -c10 -t01:00:00 -J astrotime --wrap="time singularity exec -B $NOBACKUP,/explore/nobackup/projects,/explore/nobackup/people --nv /explore/nobackup/projects/ilab/containers/astrotime-latest python /explore/nobackup/projects/ilab/ilab_testing/astrotime/workflow/baseline-cnn.py platform.project_root=/explore/nobackup/projects/ilab/ilab_testing/astrotime data.dataset_root=/explore/nobackup/projects/ilab/data/astrotime/sinusoids/nc"
```

## References

- Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
- Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores. Nonlinear Processes in Geophysics 12, 345–352 (2005).
