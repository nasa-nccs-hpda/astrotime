from logging import FileHandler

import numpy as np, tensorflow as tf
from typing import List, Optional, Dict, Type, Any
from astrotime.models.cnn_powell import SinusoidPeriodModel
from astrotime.callbacks.printers import ShapePrinter
from astrotime.loaders.sinusoid import SinusoidLoader
from astrotime.encoders.wavelet import WaveletEncoder
from astrotime.callbacks.checkpoints import CheckpointCallback
from tensorflow.compat.v1 import logging
from astrotime.util.logging import lgm

data_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/sinusoids/npz/"
results_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/results"
seq_length = 1000
epochs=1000
batch_size=64
nfeatures=5
train_dset_idx = 0
valid_dset_idx = 1
optimizer='rmsprop'
loss='mae'
model_name = f"wwz-{nfeatures}"
rank = 0
log_level = logging.INFO

lgm().init_logging( f"{results_dir}/logging", rank, log_level )
device = f"/device:GPU:{rank}" if rank >= 0 else "/CPU:0"

sinusoid_loader = SinusoidLoader(data_dir)
encoder = WaveletEncoder(device)

tdset: Dict = sinusoid_loader.get_dataset(train_dset_idx)
train_data:   tf.Tensor = encoder.encode_dset(tdset)
train_target: tf.Tensor  = tdset['target']

vdset = sinusoid_loader.get_dataset(valid_dset_idx)
valid_data:   tf.Tensor   = encoder.encode_dset(vdset)
valid_target: tf.Tensor   = vdset['target']

print( f" *** train_data {type(train_data)} train_target {type(train_target)} valid_data {type(valid_data)} valid_target {type(valid_target)} ")

print( f" *** train_data{train_data.shape.as_list()} train_target{train_target.shape.as_list()} valid_data{valid_data.shape.as_list()} valid_target{valid_target.shape.as_list()} ")

shape_printer = ShapePrinter(input_shapes=train_data.shape)
checkpointer = CheckpointCallback( model_name, f"{results_dir}/checkpoints" )
train_args: Dict[str,Any] = dict( epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[shape_printer,checkpointer], verbose=1  )

spmodel = SinusoidPeriodModel(seq_length)
spmodel.compile(optimizer=optimizer, loss=loss)
history = spmodel.fit( train_data, train_target, validation_data=(valid_data, valid_target), **train_args )



