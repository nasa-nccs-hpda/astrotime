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
from astrotime.util.env import parse_clargs, get_device
from astrotime.loaders.base import DataPreprocessor

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
refresh = False

sinusoid_loader = ncSinusoidLoader( dataset_root, dataset_files, file_size, batch_size )
encoder = ValueEncoder(device,series_length)
generator = DataPreprocessor( sinusoid_loader, encoder )
model: keras.Model = SinusoidPeriodModel()
model.compile(optimizer=optimizer, loss=loss)

shape_printer = ShapePrinter(input_shapes=generator.shape)
checkpointer = CheckpointCallback( model_name, f"{results_dir}/checkpoints" )
train_args: Dict[str,Any] = dict( epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[shape_printer,checkpointer], verbose=1)

prediction = model.predict( generator()[0] )
if refresh: print( "Refreshing model. Training from scratch.")
else: checkpointer.load_weights(model)
history = model.fit( generator.get_dataset(), **train_args )
