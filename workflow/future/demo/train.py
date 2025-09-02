import time, os, math, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tmodel

signal_index=2
expt_index=1
nepochs=5000
batch_size=64
learning_rate=0.001
dropout_frac=0.0
use_ckpt=True
loss='mse'

data = tmodel.get_demo_data()
signals = data['signals']
times = data['times']
ckp_file = tmodel.get_ckp_file( expt_index, signal_index )

# X = binary_times[signal_index].astype(np.float32)
X: np.ndarray = tmodel.get_features( times[signal_index], expt_index )
Y: np.ndarray = signals[signal_index]
validation_split: int = int(0.8*X.shape[0])

Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

optimizer = Adam( learning_rate=learning_rate, name='adam' )
tmodel = tmodel.create_small_model(X.shape[1],dropout_frac)
tmodel.compile(optimizer=optimizer, loss=loss )
if use_ckpt:
    if os.path.exists(ckp_file): tmodel.load_weights( ckp_file )
    else: print( f"Checkpoint file '{ckp_file}' not found. Training from scratch." )

ckp_args = dict( save_best_only=True, save_weights_only=True, monitor='val_loss' )
ckp_callback = ModelCheckpoint(ckp_file, **ckp_args)

t0 = time.time()
history = tmodel.fit(
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


