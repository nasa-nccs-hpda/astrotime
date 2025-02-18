import os.path

import tensorflow as tf
import keras
from typing import List, Optional, Dict, Type, Any
from astrotime.encoders.baseline import ValueEncoder
from astrotime.models.cnn_powell import SinusoidPeriodModel
from astrotime.callbacks.printers import ShapePrinter
from astrotime.loaders.sinusoid import ncSinusoidLoader
from astrotime.callbacks.checkpoints import CheckpointCallback
from argparse import Namespace
from astrotime.transforms.filters import RandomDownsample
from astrotime.util.env import parse_clargs, get_device
from astrotime.loaders.base import DataGenerator
from tensorflow.compat.v1 import logging
from astrotime.util.logging import lgm

ccustom = {}
clargs: Namespace = parse_clargs(ccustom)
device = get_device(clargs)
dataset_root=  "/explore/nobackup/projects/ilab/data/astro_sigproc/sinusoids/nc"
results_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/results"
dataset_files=  "padded_sinusoids_*.nc"
file_size= 1000
batch_size= 100
model_name = "baseline"
series_length = 2000
epochs=1000
eval_size=10
optimizer='rmsprop'
loss='mae'
refresh = True
sparsity = 0.0
log_level = logging.INFO
lgm().init_logging( f"{results_dir}/logging", log_level )

sinusoid_loader = ncSinusoidLoader( dataset_root, dataset_files, file_size, batch_size )
encoder = ValueEncoder(device,series_length)
if sparsity > 0.0: encoder.add_filters( [RandomDownsample(sparsity=sparsity)] )
generator = DataGenerator( sinusoid_loader, encoder )
model: keras.Model = SinusoidPeriodModel()
model.compile(optimizer=optimizer, loss=loss)
input_batch, target_batch = generator[0]

shape_printer = ShapePrinter(input_shapes=input_batch.shape)
checkpointer = CheckpointCallback( model_name, f"{results_dir}/checkpoints" )
train_args: Dict[str,Any] = dict( epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[shape_printer,checkpointer], verbose=1)

prediction = model.predict( input_batch )
if refresh: print( "Refreshing model. Training from scratch.")
else: checkpointer.load_weights(model)
history = model.fit( generator, **train_args )
