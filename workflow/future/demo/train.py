import time, os, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

signal_index=2
expt_index=2
nepochs=1000
batch_size=64
learning_rate=0.001
dropout_frac=0.5
use_ckpt=True

data_dir =  "/explore/nobackup/projects/ilab/data/astrotime/demo"
ckp_file = f"{data_dir}/embed_time_predict.e{expt_index}.s{signal_index}.weights.h5"
data=np.load( f'{data_dir}/jordan_data.npz',allow_pickle=True )
signals = data['signals']
times = data['times']
binary_times = data['binary_times']

# def get_features( T: np.ndarray ) -> np.ndarray:
#     for ibase, np in [ (2,12), (3,8), (5,5), (6,4), (7,3) ]:
#         for ip in range(1,np):
#             p = mod
#             f = np.mod(r,23).tolist()/23

X = binary_times[signal_index].astype(np.float32)
Y = signals[signal_index]
validation_split = int(0.8*X.shape[0])

Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

def create_small_model(dropout_frac):
    binary_times_input = tf.keras.Input(shape=(64,), name="binary_times_input")

    x = tf.keras.layers.Dense(512, activation='tanh')(binary_times_input)
    x = tf.keras.layers.Dropout(dropout_frac)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='tanh')(x)
    x = tf.keras.layers.Dropout(dropout_frac)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='tanh')(x)
    x = tf.keras.layers.Dropout(dropout_frac)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='tanh')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=binary_times_input, outputs=outputs)
    return model

optimizer = Adam( learning_rate=learning_rate, name='adam' )
model = create_small_model(dropout_frac)
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

