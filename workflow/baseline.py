import numpy as np
from astrotime.encoders.baseline import ValueEncoder
from astrotime.models.cnn_powell import SinusoidPeriodModel
from astrotime.callbacks.printers import ShapePrinter
from astrotime.loaders.sinusoid import SinusoidLoader

# data_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/sinusoids/npz/"
data_dir = "/Users/tpmaxwel/Data/astro_sigproc/sinusoids"
seq_length = 1000
epochs=1000
batch_size=64
train_dset_idx = 2
valid_dset_idx = 3
optimizer='rmsprop'
loss='mae'

sinusoid_loader = SinusoidLoader(data_dir)
encoder = ValueEncoder(seq_length)
model = SinusoidPeriodModel(seq_length)
model.compile(optimizer=optimizer, loss=loss)

tdset = sinusoid_loader.get_dataset(train_dset_idx)
train_data = encoder.encode_dset(tdset)
train_target = tdset['target']

vdset = sinusoid_loader.get_dataset(valid_dset_idx)
valid_data  = encoder.encode_dset(vdset)
valid_target = vdset['target']

shape_printer = ShapePrinter(input_shapes=train_data.shape)
train_args = dict( epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[shape_printer], verbose=1  )

history = model.fit( train_data, train_target, validation_data=(valid_data, valid_target), **train_args )
