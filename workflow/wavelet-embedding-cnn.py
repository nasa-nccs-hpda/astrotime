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

logger = tf.get_logger()
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

lgm().init_logging( f"{results_dir}/logging", rank, logging.DEBUG )
device = f"/device:GPU:{rank}" if rank >= 0 else "/CPU:0"

sinusoid_loader = SinusoidLoader(data_dir)
encoder = WaveletEncoder(device)
tdset: Dict = sinusoid_loader.get_dataset(train_dset_idx)
encoded = encoder.encode_dset(tdset)



