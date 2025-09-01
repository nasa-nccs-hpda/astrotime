import time, os, math, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from .model import create_small_model, get_demo_data, get_ckp_file, get_features

signal_index=2
expt_index=4
nepochs=1000
batch_size=64
learning_rate=0.001
dropout_frac=0.0
use_ckpt=True

data = get_demo_data()
signals = data['signals']
times = data['times']
ckp_file = get_ckp_file( expt_index, signal_index )

# X = binary_times[signal_index].astype(np.float32)
X: np.ndarray = get_features( times[signal_index] )
Y: np.ndarray = signals[signal_index]
validation_split: int = int(0.8*X.shape[0])

Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

optimizer = Adam( learning_rate=learning_rate, name='adam' )
model = create_small_model(X.shape[1],dropout_frac)
model.compile(optimizer=optimizer, loss='mae')
if use_ckpt:
    if os.path.exists(ckp_file): model.load_weights( ckp_file )
    else: print( f"Checkpoint file '{ckp_file}' not found. Training from scratch." )

ckp_args = dict( save_best_only=True, save_weights_only=True, monitor='val_loss' )
ckp_callback = ModelCheckpoint(ckp_file, **ckp_args)

t0 = time.time()
history = model.fit(
    Xtrain,
    Ytrain,
    epochs=nepochs,
    validation_data=(Xval,Yval),
    callbacks=[ckp_callback],
    batch_size=batch_size,
    shuffle=True
)
print( f"Completed training for {nepochs} epochs in {(time.time()-t0)/60:.2f} min.")
print( f"Saved checkpoint to '{ckp_file}'")


