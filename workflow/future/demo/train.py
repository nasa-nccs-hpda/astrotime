import time, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

signal_index=3
expt_index=0
nepochs=1000
use_ckpt=True
optimizer = 'rmsprop'

data_dir =  "/explore/nobackup/projects/ilab/data/astrotime/demo"
ckp_file = f"{data_dir}/embed_time_predict.e{expt_index}.s{signal_index}.weights.h5"
data=np.load( f'{data_dir}/jordan_data.npz',allow_pickle=True )
signals = data['signals']
times = data['times']
binary_times = data['binary_times']

X = binary_times[signal_index].astype(np.float32)
Y = signals[signal_index]
validation_split = int(0.8*X.shape[0])

Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    x = tf.keras.layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)

    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('leaky_relu')(x)

    return x


def create_resnet_model(n_filters_start):
    binary_times_input = tf.keras.Input(shape=(64,), name="binary_times_input")

    x = tf.keras.layers.Dense(256, activation='tanh')(binary_times_input)
    x = tf.keras.layers.Dense(256, activation='tanh')(x)
    x = tf.keras.layers.Dense(128, activation='tanh')(x)
    x = tf.keras.layers.Reshape((128,1))(x)

    nblock=0
    x = residual_block(x, n_filters_start)
    x = residual_block(x, n_filters_start+4)
    x = residual_block(x, n_filters_start+8)
    x = residual_block(x, n_filters_start+12)
    nblock+=1
    x = residual_block(x, n_filters_start+(16*nblock), stride=2)
    x = residual_block(x, n_filters_start+(16*nblock+4))
    x = residual_block(x, n_filters_start+(16*nblock+8))
    x = residual_block(x, n_filters_start+(16*nblock+12))
    nblock+=1
    x = residual_block(x, n_filters_start+(16*nblock), stride=2)
    x = residual_block(x, n_filters_start+(16*nblock+4))
    x = residual_block(x, n_filters_start+(16*nblock+8))
    x = residual_block(x, n_filters_start+(16*nblock+12))
    nblock+=1
    x = residual_block(x, n_filters_start+(16*nblock), stride=2)
    x = residual_block(x, n_filters_start+(16*nblock+4))
    x = residual_block(x, n_filters_start+(16*nblock+8))
    x = residual_block(x, n_filters_start+(16*nblock+12))
    nblock+=1
    x = residual_block(x, n_filters_start+(16*nblock), stride=2)
    x = residual_block(x, n_filters_start+(16*nblock+4))
    x = residual_block(x, n_filters_start+(16*nblock+8))
    x = residual_block(x, n_filters_start+(16*nblock+12))
    nblock+=1
    x = residual_block(x, n_filters_start+(16*nblock), stride=2)
    x = residual_block(x, n_filters_start+(16*nblock+4))
    x = residual_block(x, n_filters_start+(16*nblock+8))
    x = residual_block(x, n_filters_start+(16*nblock+12))
    nblock+=1
    x = residual_block(x, n_filters_start+(16*nblock), stride=2)
    x = residual_block(x, n_filters_start+(16*nblock+4))
    x = residual_block(x, n_filters_start+(16*nblock+8))
    x = residual_block(x, n_filters_start+(16*nblock+12))

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='linear')(x)
    x = tf.keras.layers.Dense(512, activation='linear')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=binary_times_input, outputs=outputs)
    return model

model = create_resnet_model(32)
model.compile(optimizer=optimizer, loss='mae')
if use_ckpt: model.load_weights( ckp_file )

ckp_args = dict( save_best_only=True, save_weights_only=True, monitor='val_loss' )
ckp_callback = ModelCheckpoint(ckp_file, **ckp_args)

t0 = time.time()
history = model.fit(
    Xtrain,
    Ytrain,
    epochs=nepochs,
    validation_data=(Xval,Yval),
    callbacks=[ckp_callback],
    batch_size=256,
    shuffle=True
)
print( f"Completed training for {nepochs} epochs in {(time.time()-t0)/60:.2f} min.")

