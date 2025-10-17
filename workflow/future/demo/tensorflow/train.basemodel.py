import numpy as np
import tmodel
import tensorflow as tf
from tensorflow import keras
from base_model import float_to_binary_array_not_IEEE, create_dense_model

data=tmodel.get_demo_data()
signals = data['signals']
times = data['times']
signal=2
ninstances = 4

X = times[signal].copy()
X = X/X.max()
X = np.array([float_to_binary_array_not_IEEE(X[i]) for i in range(len(X))])
Y = signals[signal]
validation_split = int(0.8*X.shape[0])

Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

strategy = tf.distribute.MirroredStrategy()

for ti in range(ninstances):
    with strategy.scope():
        model = create_dense_model(dropout_frac=0.5,n_streams=20)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=.01), loss='mae')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(f"{tmodel.data_dir}/base_model_{signal}_{ti}.weights.h5", save_best_only=True, save_weights_only=True, monitor='val_loss')

    history = model.fit(
        Xtrain,
        Ytrain,
        epochs=10000,
        validation_data=(Xval,Yval),
        callbacks=[checkpoint_callback],
        batch_size=512,
        shuffle=True
    )


