import time, os, math, argparse, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tmodel

parser = argparse.ArgumentParser(
                    prog='timehascome',
                    usage='python train.py --help',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    description='Trains time-aware model on demo data.')

parser.add_argument('-s', '--signal',     type=int, default=2)
parser.add_argument('-e', '--experiment', type=int, default=1)
parser.add_argument('-n', '--nepochs',    type=int, default=500)
parser.add_argument('-r', '--refresh',    action='store_true')
args = parser.parse_args()

print( f"\nRunning with args: {args}\n")

signal_index=args.signal
expt_index=args.experiment
nepochs=args.nepochs
batch_size=256
dropout_frac=0.5
refresh=args.refresh
loss='mae'

data = tmodel.get_demo_data()
signals = data['signals']
times = data['times']
ckp_file = tmodel.get_ckp_file( expt_index, signal_index )
if refresh and os.path.exists(ckp_file): os.remove(ckp_file)

X: np.ndarray = tmodel.get_features( times[signal_index], expt_index )
Y: np.ndarray = signals[signal_index]
validation_split: int = int(0.8*X.shape[0])
print( f"X.shape={X.shape}, Y.shape={Y.shape}")
# Y = tmodel.tnorm(Y)

Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

tmodel = tmodel.create_small_model(X.shape[1],dropout_frac)
tmodel.compile(optimizer='rmsprop', loss='mae')

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


