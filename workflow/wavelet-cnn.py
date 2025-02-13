import numpy as np
from typing import List, Optional, Dict, Type, Any
from astrotime.encoders.wavelet import WaveletEncoder
from astrotime.models.cnn_powell import SinusoidPeriodModel
from astrotime.callbacks.printers import ShapePrinter
from astrotime.loaders.wwz import WaveletLoader

data_dir = "/explore/nobackup/projects/ilab/projects/fusion/cache/encodings/"
seq_length = 1000
epochs=1000
batch_size=64
train_dset_idx = 0
valid_dset_idx = 1
optimizer='rmsprop'
loss='mae'

sinusoid_loader = WaveletLoader(data_dir)
encoder = WaveletEncoder(seq_length)
model = SinusoidPeriodModel(seq_length)
model.compile(optimizer=optimizer, loss=loss)

tdset: Dict[ str, np.ndarray] = sinusoid_loader.get_dataset(train_dset_idx)
train_data:   np.ndarray  = encoder.encode_dset(tdset)
train_target: np.ndarray  = tdset['target']

vdset = sinusoid_loader.get_dataset(valid_dset_idx)
valid_data:   np.ndarray  = encoder.encode_dset(vdset)
valid_target: np.ndarray  = vdset['target']

shape_printer = ShapePrinter(input_shapes=train_data.shape)
train_args: Dict[str,Any] = dict( epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[shape_printer], verbose=1  )

history = model.fit( train_data, train_target, validation_data=(valid_data, valid_target), **train_args )
