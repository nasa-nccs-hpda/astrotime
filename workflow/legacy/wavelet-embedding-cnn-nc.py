import os, numpy as np, tensorflow as tf, keras
from typing import List, Optional, Dict, Type, Any
from astrotime.models.cnn_baseline import get_model
from astrotime.callbacks.printers import ShapePrinter
from astrotime.loaders.sinusoid import ncSinusoidLoader
from astrotime.encoders.wavelet import WaveletEncoder
from astrotime.callbacks.checkpoints import CheckpointCallback
from astrotime.loaders.base import DataGenerator
from tensorflow.compat.v1 import logging
from astrotime.util.logging import lgm
from argparse import Namespace
from astrotime.util.env import parse_clargs, get_device
from astrotime.transforms.filters import RandomDownsample

ccustom = {}
clargs: Namespace = parse_clargs(ccustom)
device = get_device(clargs)

# dataset_root=  "/explore/nobackup/projects/ilab/data/astro_sigproc/sinusoids/nc"
# results_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/results"

dataset_root  = "/Users/tpmaxwel/Data/astro_sigproc/sinusoids/nc"
results_dir   = "/Users/tpmaxwel/Data/astro_sigproc/results"

dataset_files=  "padded_sinusoids_*.nc"
file_size= 1000
batch_size= 50

epochs=1000

optimizer='rmsprop'
loss='mae'
model_name = f"wwz-{nfeatures}"
log_level = logging.INFO
lgm().init_logging( f"{results_dir}/logging", log_level )
refresh = False

sinusoid_loader = ncSinusoidLoader( dataset_root, dataset_files, file_size, batch_size )

series_length = 2000
nfeatures=2
nfreq: int = 2000
fbounds = (0.1,10.0)
fscale = "log"
sparsity = 0.0
max_series_length = 6200

encoder = WaveletEncoder( device, series_length, nfreq, fbounds, fscale, nfeatures, int(max_series_length*(1-sparsity)) )
if sparsity > 0.0:
	encoder.add_filters( [RandomDownsample(sparsity=sparsity)] )
generator = DataGenerator( sinusoid_loader, encoder )
sample_input, sample_target = generator[0]

shape_printer = ShapePrinter(input_shapes=sample_input.shape)
checkpointer = CheckpointCallback( model_name, f"{results_dir}/checkpoints" )
train_args: Dict[str,Any] = dict( epochs=epochs, batch_size=batch_size, shuffle=False, callbacks=[shape_printer,checkpointer], verbose=1)

spmodel: keras.Model = get_model( sample_input.shape, optimizer=optimizer, loss=loss )
if refresh: print( "Refreshing model. Training from scratch.")
else: checkpointer.load_weights(spmodel)

history = spmodel.fit( generator, **train_args )



