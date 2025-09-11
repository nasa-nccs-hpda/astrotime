import numpy as np, os, time
import tensorflow as tf
from tensorflow import keras
from argparse import Namespace
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tmodel, argparse
default_data_dir = "/explore/nobackup/projects/ilab/data/astrotime/demo"

parser = argparse.ArgumentParser( prog='timehascome', usage='python train.py --help', description='Trains time-aware CNN on demo data.')
parser.add_argument('-s',  '--signal',        type=int, default=2)
parser.add_argument('-f',  '--feature_type',  type=int, default=1)
parser.add_argument('-ne', '--nepochs',       type=int, default=2000)
parser.add_argument('-nf', '--nfeatures',     type=int, default=32)
parser.add_argument('-bs', '--batch_size',    type=int, default=512)
parser.add_argument('-l',  '--loss',          type=str, default="mae")
parser.add_argument('-ns', '--nstreams',      type=int, default=10)
parser.add_argument('-r',  '--refresh',       action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
parser.add_argument('-pf', '--minp_factor',   type=float, default=2.0)
parser.add_argument('-do', '--dropout_frac',  type=float, default=0.5)
parser.add_argument('-dd',  '--data_dir',     type=str, default=default_data_dir)
args: Namespace = tmodel.parse_args(parser)

signal_index=args.signal
feature_type=args.feature_type

data=tmodel.get_demo_data()
signals = data['signals']
times = data['times']
X: np.ndarray = tmodel.get_features( times[signal_index], feature_type, args )
Y = signals[signal_index]

validation_split = int(0.8*X.shape[0])
Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")
with strategy.scope():
    model = tmodel.create_streams_model( X.shape[1], dropout_frac=args.dropout_frac, n_streams=args.nstreams )
    model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=args.learning_rate ), loss=args.loss )

ckp_file = tmodel.get_ckp_file( args )
if args.refresh and os.path.exists(ckp_file): os.remove(ckp_file)
if os.path.exists(ckp_file): model.load_weights(ckp_file)
else: print( f"Checkpoint file '{ckp_file}' not found. Training from scratch." )
ckp_args = dict( save_best_only=True, save_weights_only=True, monitor='val_loss' )
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( ckp_file, **ckp_args )

t0 = time.time()
history = model.fit(
    Xtrain,
    Ytrain,
    epochs=args.nepochs,
    validation_data=(Xval,Yval),
    callbacks=[checkpoint_callback],
    batch_size=args.batch_size,
    shuffle=True
)
print( f"Completed training for {args.nepochs} epochs in {(time.time()-t0)/60:.2f} min.")
print( f"Saved checkpoint to '{ckp_file}'  ")
