import numpy as np
import pandas as pd
import csv 
import time


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Lambda, dot, Activation, concatenate, Reshape
from tensorflow.keras.layers import Layer

class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        attention_vec_reshape = Reshape((128, 1), name='attention_vec_reshape')(attention_vector)
        return attention_vec_reshape


paddeddata_csv = pd.read_csv("paddeddata91.csv") # 讀資料進來
paddeddata_array = np.array(paddeddata_csv) # 轉為矩陣
# -------------------------------------------------------------------------------------------------------------------
# 平均值正規化 [-1, 1]
data_min = np.min(paddeddata_array[:, 1:4], axis=0)
data_max = np.max(paddeddata_array[:, 1:4], axis=0)
data_mean = np.mean(paddeddata_array[:, 1:4], axis=0)
print(data_min)
print(data_max)
print(data_mean)
paddeddata_array_norm = paddeddata_array
paddeddata_array_norm[:, 1:4] = (paddeddata_array[:, 1:4]-data_mean)/(data_max-data_min)
# print(paddeddata_array[])
count = 1
the_first_nonzero = 0
the_last_nonzero = 0
n = 0
def GenDataset(inputdata, starttime, lasttime, lookback, delay, samp_list, targ_list):
    for i in range(lasttime-starttime+1):
        input_raws = np.arange(starttime, starttime+lookback)
        output_raws = np.arange(starttime+lookback, starttime+lookback+delay)
        samp_list.append(inputdata[input_raws, 1:4])
        targ_list.append(inputdata[output_raws, 1:4])
    return samp_list, targ_list
sample_list = []
target_list = []
while 1:
    # for n in range(the_first_nonzero, len(paddeddata_array)):
    if paddeddata_array_norm[n, 4] == 0:
        the_last_nonzero = n-1
        count = 1
        print("creat from {} to {}".format(the_first_nonzero, the_last_nonzero))
        GenDataset(inputdata=paddeddata_array_norm, starttime=the_first_nonzero, lasttime=the_last_nonzero, lookback=288, delay=24, samp_list=sample_list, targ_list=target_list)
        # check how many zero in next row ~~
        for p in range(n+1, len(paddeddata_array_norm)):
            if paddeddata_array_norm[p, 4] == 0: 
                count += 1
            else:
                the_first_nonzero = the_last_nonzero + count + 1  
                n = the_first_nonzero
                break
    n += 1 
    if n == len(paddeddata_array_norm):
        break
sample_arr = np.array(sample_list)
target_arr = np.array(target_list)
print(sample_arr.shape)
print(target_arr.shape)
# print(target_arr)
# # # # # # # # # # # # # # # # # # # # # # # # # 
# -------------------------------------------------------------------------------------------------------------------
# train test split
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(sample_arr, target_arr[:, :, 0], test_size=0.3)
x_train = sample_arr[:12651]
x_test = sample_arr[12651:]
y_train = target_arr[:12651, :, 0]
y_test = target_arr[12651:, :, 0]
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()

model.add((LSTM(128,
                input_shape=(288, 3), 
                return_sequences=True
                )))

model.add((LSTM(128, 
                return_sequences=True
                )))

model.add(Attention(name='attention_weight'))

model.add((LSTM(24, 
                return_sequences=False
                )))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(24, activation='relu'))
# -----------------------------------------------------
model.summary()
model.compile(optimizer=RMSprop(), loss='mae')
# checkpoint
# filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True,
                             mode='auto')
callbacks_list = [checkpoint]

history = model.fit(x_train, y_train,
                    epochs=70,
                    batch_size=1024,
                    callbacks=callbacks_list,
                    validation_split=0.3)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()


model.load_weights("weights.best.hdf5")
test_results = model.evaluate(x_test, y_test, batch_size=4096)
print(test_results)

predictions = model.predict(x_test, verbose=1, batch_size=128)
print(predictions.shape)

test_MAE = np.mean(abs(predictions[1069, :]-y_test[1069, :]))
print(test_MAE)
# plt.figure()
example_test = np.concatenate((x_test[1069, :, 0].reshape(-1, 1),y_test[1069, :].reshape(-1, 1))) 
print(example_test.shape)
plt.subplot(1, 2, 2)
x1 = np.arange(1, 288+24+1, 1)
x2 = np.arange(289, 288+24+1, 1)
plt.plot(x1, example_test, 'ro', ms=0.5)
plt.plot(x2, predictions[1069, :], 'bs', ms=0.5)
plt.legend()

# plotfigure = 6
# for result_time in range(plotfigure):
#     pose = 69+result_time*100
#     example_test = np.concatenate((x_test[pose, :, 0].reshape(-1, 1),y_test[pose, :].reshape(-1, 1))) 
#     print(example_test.shape)

#     test_MAE = np.mean(abs(predictions[pose, :]-y_test[pose, :]))
#     print(test_MAE)
#     if result_time<4:
#         plt.subplot(2, 3, result_time+1)
#         x1 = np.arange(1, 288+24+1, 1)
#         x2 = np.arange(289, 288+24+1, 1)
#         plt.plot(x1, example_test, 'ro', ms=0.5)
#         plt.plot(x2, predictions[pose, :], 'bs', ms=0.5)
#     if result_time>=4:
#         plt.subplot(2, 3, result_time+1)
#         x1 = np.arange(1, 288+24+1, 1)
#         x2 = np.arange(289, 288+24+1, 1)
#         plt.plot(x1, example_test, 'ro', ms=0.5)
#         plt.plot(x2, predictions[pose, :], 'bs', ms=0.5)

plt.show()