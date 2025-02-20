import tensorflow as tf
from typing import Dict, Any
from models.SCRAP.cnn_powell import SinusoidPeriodModel
from astrotime.callbacks.printers import ShapePrinter
from astrotime.loaders.sinusoid import SinusoidLoader
from astrotime.encoders.wavelet import WaveletEncoder
from astrotime.callbacks.checkpoints import CheckpointCallback
from tensorflow.compat.v1 import logging
from astrotime.util.logging import lgm
from argparse import Namespace
from astrotime.util.env import parse_clargs, get_device
from astrotime.transforms.filters import RandomDownsample

ccustom = {}
clargs: Namespace = parse_clargs(ccustom)
device = get_device(clargs)

data_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/sinusoids/npz/"
results_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/results"
series_length = 2000
epochs=1000
batch_size=32
nfeatures=5
nfreq: int = 2000
fbounds = (0.1,10.0)
fscale = "log"
sparsity = 0.5
max_series_length = 6000
train_dset_idx = 0
valid_dset_idx = 1
optimizer='rmsprop'
loss='mae'
model_name = f"wwz-{nfeatures}"
log_level = logging.INFO
lgm().init_logging( f"{results_dir}/logging", log_level )
refresh = False

sinusoid_loader = SinusoidLoader(data_dir)
encoder = WaveletEncoder(device, series_length, nfreq, fbounds, fscale, nfeatures, int(max_series_length*(1-sparsity)) )
encoder.add_filters( [RandomDownsample(sparsity=sparsity)] )

tdset: Dict = sinusoid_loader.get_dataset(train_dset_idx)
tX, tY = encoder.encode_dset(tdset)
train_target: tf.Tensor  = tdset['target']

vdset = sinusoid_loader.get_dataset(valid_dset_idx)
vX, vY   = encoder.encode_dset(vdset)
valid_target: tf.Tensor   = vdset['target']

shape_printer = ShapePrinter(input_shapes=tY.shape)
checkpointer = CheckpointCallback( model_name, f"{results_dir}/checkpoints" )
train_args: Dict[str,Any] = dict( epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[shape_printer,checkpointer], verbose=1  )

spmodel = SinusoidPeriodModel()
spmodel.compile(optimizer=optimizer, loss=loss)
if refresh: print( "Refreshing model. Training from scratch.")
else: checkpointer.load_weights(spmodel)

history = spmodel.fit( tY, train_target, validation_data=(vY, valid_target), **train_args )



