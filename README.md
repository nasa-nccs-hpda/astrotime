## Astrotime

#### Machine learning methods for irregularly spaced time series

### Conda environment

* On Adapt load modules: gcc/12.1.0, nvidia/12.1
* If mamba is not available, install [miniforge](https://github.com/conda-forge/miniforge) (or load mamba module)
* Execute the following to set up a conda environment for astrotime:

###### Torch Environment (Current)

    >   * mamba create -n astrotime.pt ninja python=3.10
    >   * mamba activate astrotime
    >   * pip install torch jupyterlab==4.0.13 ipywidgets==7.8.4 cuda-python jupyterlab_widgets ipykernel==6.29 ipympl ipython==8.26 xarray netCDF4
    >   * pip install torchcde torchdiffeq hydra-core rich  scikit-learn 

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
    *   _Baseline_ (**.workflow/train-baseline-cnn.py**):  This workflow runs the baseline CNN (developed by Brian Powell) which takes only timeseries value data as input.
    *   _WWZ_ (**.workflow/train-wwz-cnn.py**): This workflow runs the same baseline CNN operating on a weighted wavelet z-transform, which enfolds both the time and value data from the timeseries. 
The *_small versions execute the workflows on a subset (1/10) of the full training dataset.
The workflows save checkpoint files at the end of each epoch.  By default the model is initialized with any existing checkpoint file at the begining of script execution.  To
execute the script with a new set of checkpoints (while keeping the old ones), create a new script with a different value of the *version* parameter.  

### Configuration

The workflows are configured using [hydra](https://hydra.cc/docs/intro/).
* All hydra yaml configuration files are found under **.config**.
* The workflow configurations can be modified at runtime as [supported by hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).
* For example, the following command runs the baseline workflow on gpu 3 with random initialization (i.e. ignoring & overwriting any existing checkpoints):
    >   python workflow/train-baseline-cnn.py platform.gpu=3 train.refresh_state=True
* To run validation (no training), execute:
    >   python workflow/train-baseline-cnn.py train.mode=valid platform.gpu=0

#### Configuration Parameters

Here is a partial list of configuration parameters with typical values.  Their values are configured in the hydra yaml files and reconfigurable on the command line:

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