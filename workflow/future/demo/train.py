import time, os, math, argparse, numpy as np
from argparse import Namespace
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tmodel

parser = argparse.ArgumentParser( prog='timehascome', usage='python train.py --help', description='Trains time-aware CNN on demo data.')
parser.add_argument('-s',  '--signal',     type=int, default=2)
parser.add_argument('-e',  '--experiment', type=int, default=1)
parser.add_argument('-ne', '--nepochs',    type=int, default=2000)
parser.add_argument('-nf', '--nfeatures',  type=int, default=64)
parser.add_argument('-bs', '--batch_size', type=int, default=256)
parser.add_argument('-l',  '--loss',       type=str, default="mae")
parser.add_argument('-r',  '--refresh',    action='store_true')
parser.add_argument('-lr', '--learning_rate',  type=float, default=0.001)
parser.add_argument('-pf', '--minp_factor',  type=float, default=1.0)
parser.add_argument('-do', '--dropout',  type=float, default=0.0)

args: Namespace = tmodel.parse_args(parser)

signal_index=args.signal
expt_index=args.experiment
nepochs=args.nepochs
learning_rate=args.learning_rate
batch_size=args.batch_size
dropout_frac=args.dropout
refresh=args.refresh
loss=args.loss

data = tmodel.get_demo_data()
signals = data['signals']
times = data['times']
ckp_file = tmodel.get_ckp_file( expt_index, signal_index )
if refresh and os.path.exists(ckp_file): os.remove(ckp_file)

X: np.ndarray = tmodel.get_features( times[signal_index], expt_index, args )
Y: np.ndarray = signals[signal_index]
validation_split: int = int(0.8*X.shape[0])

Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

small_model = tmodel.create_small_model(X.shape[1],dropout_frac)
small_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mae')
if os.path.exists(ckp_file): small_model.load_weights( ckp_file )
else: print( f"Checkpoint file '{ckp_file}' not found. Training from scratch." )
ckp_args = dict( save_best_only=True, save_weights_only=True, monitor='val_loss' )
ckp_callback = ModelCheckpoint(ckp_file, **ckp_args)

t0 = time.time()
history = small_model.fit(
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


