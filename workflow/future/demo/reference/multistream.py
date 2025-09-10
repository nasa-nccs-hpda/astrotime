
import numpy as np
import tensorflow as tf
from tensorflow import keras
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt

def float_to_binary_precise(num, places=64):
    getcontext().prec = places
    decimal_num = Decimal(str(num))
    integer_part = int(decimal_num)
    fractional_part = decimal_num - integer_part

    fractional_binary = ""
    for _ in range(places):
        fractional_part *= 2
        bit = int(fractional_part)
        fractional_binary += str(bit)
        fractional_part -= bit

    return fractional_binary

def float_to_binary_array_not_IEEE(x):
    binary_str=float_to_binary_precise(x,places=64)
    return np.array([int(bit) for bit in binary_str], dtype=np.float64)


class FirstLastElementConstraint(tf.keras.constraints.Constraint):
    def __call__(self, w):
        input_dim = w.shape[-2]
        output_dim = w.shape[-1]
        mask = tf.concat([
            tf.ones((1, input_dim, output_dim)),
            tf.zeros((tf.shape(w)[0] - 2, input_dim, output_dim)),
            tf.ones((1, input_dim, output_dim))
        ], axis=0)
        return w * mask



def create_dense_model(dropout_frac, n_streams):
    binary_times_input = tf.keras.Input(shape=(64,), name="binary_times_input")

    def apply_relpos(xx):
        x = tf.keras.layers.Dense(512, activation='elu')(xx)
        x = tf.keras.layers.Dropout(dropout_frac)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation='elu')(x)
        x = tf.keras.layers.Dropout(dropout_frac)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation='elu')(x)
        x = tf.keras.layers.Dropout(dropout_frac)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation='elu')(x)
        x = tf.keras.layers.Dropout(dropout_frac)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation='elu')(x)
        x = tf.keras.layers.Dropout(dropout_frac)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation='elu')(x)
        return x


    streams = [apply_relpos(binary_times_input) for i in range(n_streams)]

    x = tf.keras.layers.Concatenate(axis=-1)(streams)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_frac)(x)
    x = tf.keras.layers.Dense(512, activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=binary_times_input, outputs=outputs)
    return model

data_dir =  "/explore/nobackup/projects/ilab/data/astrotime/demo"
data=np.load( f'{data_dir}/jordan_data.npz',allow_pickle=True )
signals = data['signals']
times = data['times']

# each index is a different time series... pick an index from 0-99
index=2

X = times[index].copy()
# Normalize
X = X/X.max()
# Use new base 2 embedding
X = np.array([float_to_binary_array_not_IEEE(X[i]) for i in range(len(X))])
Y = signals[index]
validation_split = int(0.8*X.shape[0])

# separate into train and test
Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Wrap the model creation and compilation in the strategy's scope
with strategy.scope():
    model = create_dense_model(dropout_frac=0.5,n_streams=20)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=.01), loss='mae')

# save the weights at the lowest val_loss
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("embed_time_predict_dense.weights.h5", save_best_only=True, save_weights_only=True, monitor='val_loss')

# train
history = model.fit(
    Xtrain,
    Ytrain,
    epochs=10000,
    validation_data=(Xval,Yval),
    callbacks=[checkpoint_callback],
    batch_size=512,
    shuffle=True
)


# plot

model.load_weights('embed_time_predict_dense.weights.h5')
p0=model.predict(Xtrain,batch_size=256)
p1=model.predict(Xval,batch_size=256)

plt.figure(figsize=(15,5))
plt.plot(times[index],Y,label='truth')
plt.plot(times[index][:validation_split],p0[:,0],label='train prediction')
plt.plot(times[index][validation_split:],p1[:,0],label='val prediction')


plt.savefig('timepred37.png')
plt.close()
