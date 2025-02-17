import os.path

import tensorflow as tf
from typing import List, Optional, Dict, Type, Any
from astrotime.encoders.baseline import ValueEncoder
from astrotime.models.cnn_powell import SinusoidPeriodModel
from astrotime.callbacks.printers import ShapePrinter
from astrotime.loaders.sinusoid import SinusoidLoader
from astrotime.callbacks.checkpoints import CheckpointCallback
from argparse import Namespace
from astrotime.util.env import parse_clargs, get_device

ccustom = {}
clargs: Namespace = parse_clargs(ccustom)
device = get_device(clargs)

data_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/sinusoids/npz/"
results_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/results"
model_name = "baseline"
# data_dir = "/Users/tpmaxwel/Data/astro_sigproc/sinusoids"
series_length = 2000
epochs=1000
batch_size=32
train_dset_idx = 0
valid_dset_idx = 1
optimizer='rmsprop'
loss='mae'
refresh = False

sinusoid_loader = SinusoidLoader(data_dir)
encoder = ValueEncoder(device,series_length)
model = SinusoidPeriodModel()
model.compile(optimizer=optimizer, loss=loss)

tdset: Dict = sinusoid_loader.get_dataset(train_dset_idx)
tX, tY = encoder.encode_dset(tdset)
train_target: tf.Tensor  = tdset['target']

vdset = sinusoid_loader.get_dataset(valid_dset_idx)
vX, vY   = encoder.encode_dset(vdset)
valid_target: tf.Tensor   = vdset['target']

shape_printer = ShapePrinter(input_shapes=tY.shape)
checkpointer = CheckpointCallback( model_name, f"{results_dir}/checkpoints" )
train_args: Dict[str,Any] = dict( epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[shape_printer,checkpointer], verbose=1  )

if not refresh and os.path.exists(checkpointer.filepath):
	model.load_weights(checkpointer.filepath)
history = model.fit( tY, train_target, validation_data=(vY, valid_target), **train_args )
