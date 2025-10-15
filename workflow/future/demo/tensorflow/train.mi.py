import numpy as np, os, time, shutil
import tensorflow as tf
from tensorflow import keras
from argparse import Namespace
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tmodel, argparse
default_data_dir = "/explore/nobackup/projects/ilab/data/astrotime/demo"
def intlist(arg:str): return list(map(int, arg.split(",")))

parser = argparse.ArgumentParser( prog='timehascome', usage='python train.py --help', description='Trains time-aware CNN on demo data.')
parser.add_argument('-s',  '--signal',        type=int, default=2)
parser.add_argument('-f',  '--feature_type',  type=int, default=0)
parser.add_argument('-ne', '--nepochs',       type=int, default=2000)
parser.add_argument('-nf', '--nfeatures',     type=int, default=32)
parser.add_argument('-rs', '--reduction_size', type=int, default=0)
parser.add_argument('-ni', '--ninstances',    type=int, default=8)
parser.add_argument('-ner', '--n_epoch_ranges', type=int, default=5)
parser.add_argument('-ser', '--start_epoch_ranges', type=int, default=0)
parser.add_argument('-el', '--expt_label',    type=str, default='ti')
parser.add_argument('-bs', '--batch_size',    type=int, default=512)
parser.add_argument('-l',  '--loss',          type=str, default="mae")
parser.add_argument('-ns', '--nstreams',      type=int, default=10)
parser.add_argument('-sw', '--smooth_win',    type=int, default=0)
parser.add_argument('-fc', '--feature_class', type=str, default='' )
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
parser.add_argument('-pf', '--minp_factor',   type=float, default=2.0)
parser.add_argument('-do', '--dropout_frac',  type=float, default=0.5)
parser.add_argument('-dd',  '--data_dir',     type=str, default=default_data_dir)
parser.add_argument('-dv',  '--devices',      type=intlist, default="0")
parser.add_argument('-us',  '--upscale',      type=int, default=0 )
parser.add_argument('-ds',  '--downscale',    type=int, default=0 )
args: Namespace = parser.parse_args()

data=tmodel.get_demo_data()
signals = data['signals']
times = data['times']
T: np.ndarray = times[args.signal].copy()
Y: np.ndarray = signals[args.signal]
validation_split = int(0.8 * Y.shape[0])
batches_per_epoch = validation_split // args.batch_size
Ytrain = Y[:validation_split]
Yval = Y[validation_split:]

X: np.ndarray = tmodel.get_features(T, args)
Xtrain = X[:validation_split]
Xval = X[validation_split:]

tmodel.save_args(args)
strategy = tf.distribute.MirroredStrategy([f"GPU:{i}" for i in args.devices])


for epoch_range_idx in range(args.start_epoch_ranges,args.n_epoch_ranges):

    for train_instance in range(args.ninstances):
            print(f"EXEC: epoch_range_idx={epoch_range_idx}, train_instance={train_instance}")

            with strategy.scope():
                model = tmodel.create_streams_model( X.shape[1], dropout_frac=args.dropout_frac, n_streams=args.nstreams, reduction_size=args.reduction_size )
                model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=args.learning_rate ), loss=args.loss )

            ctype = f"{args.expt_label}{epoch_range_idx*args.nepochs}_{train_instance}"
            if args.reduction_size>0: ctype += f"_rs{args.reduction_size}"
            ckp_file = tmodel.get_ckp_file( args, ctype )
            if os.path.exists(ckp_file): os.remove(ckp_file)
            if epoch_range_idx>0:
                base_ckp_file = tmodel.get_ckp_file(args, f"{args.expt_label}{(epoch_range_idx-1) * args.nepochs}_{train_instance}")
                shutil.copyfile(base_ckp_file, ckp_file)

            if os.path.exists(ckp_file):
                print(f"Loading checkpoint file '{ckp_file}'")
                model.load_weights(ckp_file)
            else:
                print( f"Checkpoint file '{ckp_file}' not found. Training from scratch." )
            ckp_callback_latest = tf.keras.callbacks.ModelCheckpoint( ckp_file, save_freq=10*batches_per_epoch, save_weights_only=True )

            t0 = time.time()
            print( f"Fit-{args.expt_label} Instance {train_instance}.{epoch_range_idx}: Xtrain{Xtrain.shape} Ytrain{Ytrain.shape} Xval{Xval.shape} Yval{Yval.shape} T{T.shape} X{X.shape} Y{Y.shape} " )
            history = model.fit(
                Xtrain,
                Ytrain,
                epochs=args.nepochs,
                validation_data=(Xval,Yval),
                callbacks=[ckp_callback_latest],
                batch_size=args.batch_size,
                shuffle=True
            )
            print( f"Completed training({train_instance}.{epoch_range_idx}) for {args.nepochs} epochs in {(time.time()-t0)/60:.2f} min.")
            print( f"Saving checkpoints to  '{ckp_file}' ")
