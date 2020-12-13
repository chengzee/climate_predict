import numpy as np
import pandas as pd
import csv 
import time
import matplotlib.pyplot as plt


import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, LSTM, Dense, Lambda, dot, Activation, concatenate, Reshape
from keras.layers import Layer
import keras.backend as K

# 參數設定------------------------------------------------------------------------------------------
count = 1
the_first_nonzero = 0
the_last_nonzero = 0
n = 0
_lookback = 288*3
_delay = 12*6
sample_list = []
target_list = []
train_size = 0.7
# neurons = [64, 128, 256, 512]

neurons = [64, 128, 256]
# A_layers = 4
# B_layers = 4
A_layers = 1
B_layers = 1
test_times = 1
_epochs = 10
_batch_size = 1024

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
        attention_vector = Dense(_delay, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        # attention_vec_reshape = Reshape((_delay, 1), name='attention_vec_reshape')(attention_vector)
        # return attention_vec_reshape
        return attention_vector


# # 定義 attention 機制 (return_sequence=True)
# class attention(Layer):
#     def __init__(self,**kwargs):
#         super(attention,self).__init__(**kwargs)
#     def build(self,input_shape):
#         self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
#         self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
#         super(attention, self).build(input_shape)
#     def call(self,x):
#         et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
#         at=K.softmax(et)
#         at=K.expand_dims(at,axis=-1)
#         print(at)
#         output=x*at
#         return K.sum(output,axis=1, keepdims=True)
#     def compute_output_shape(self,input_shape):
#         return (input_shape)
#     def get_config(self):
#         return super(attention,self).get_config()


paddeddata_csv = pd.read_csv("paddeddata1.csv") # 讀資料進來
paddeddata_array = np.array(paddeddata_csv) # 轉為矩陣
# -------------------------------------------------------------------------------------------------------------------
# 最小最大值正規化 [0, 1]
data_min = np.min(paddeddata_array[:, 1:4], axis=0)
data_max = np.max(paddeddata_array[:, 1:4], axis=0)
# data_mean = np.mean(paddeddata_array[:, 1:4], axis=0)
print("data_min:{}".format(data_min))
print("data_max:{}".format(data_max))
# # print("data_mean:{}".format(data_mean))
paddeddata_array_norm = paddeddata_array
paddeddata_array_norm[:, 1:4] = (paddeddata_array[:, 1:4]-data_min)/(data_max-data_min)
# print(paddeddata_array[])

# 參數設定------------------------------------------------------------------------------------------
def GenDataset(inputdata, starttime, lasttime, lookback, delay, samp_list, targ_list):
    for i in range(lasttime-starttime+1):
        input_raws = np.arange(i+starttime, i+starttime+lookback)
        output_raws = np.arange(i+starttime+lookback, i+starttime+lookback+delay)
        samp_list.append(inputdata[input_raws, 1:4])
        targ_list.append(inputdata[output_raws, 1:4])
    return samp_list, targ_list
while 1:
    if paddeddata_array_norm[n, 4] == 0:
        the_last_nonzero = n-1
        count = 1
        print("creat from {} to {}".format(the_first_nonzero, the_last_nonzero))
        GenDataset(inputdata=paddeddata_array_norm, starttime=the_first_nonzero, lasttime=the_last_nonzero, lookback=_lookback, delay=_delay, samp_list=sample_list, targ_list=target_list)
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
print("sample_arr.shape:{}".format(sample_arr.shape))
print("target_arr.shape:{}".format(target_arr.shape))

# # # # # # # # # # # # # # # # # # # # # # # # # 
# -------------------------------------------------------------------------------------------------------------------
# train test split
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(sample_arr, target_arr[:, :, 0], test_size=0.3)
print("len(sample_arr):{}".format(len(sample_arr)))
print("len(sample_arr)*train_size:{}".format(len(sample_arr)*train_size))
x_train = sample_arr[:int(len(sample_arr)*train_size)]
x_test = sample_arr[int(len(sample_arr)*train_size):]
y_train = target_arr[:int(len(sample_arr)*train_size), :, 0]
y_test = target_arr[int(len(sample_arr)*train_size):, :, 0]
print("x_train.shape:{}".format(x_train.shape))
print("x_test.shape:{}".format(x_test.shape))
print("y_train.shape:{}".format(y_train.shape))
print("y_test.shape:{}".format(y_test.shape))

# with open("{}LSTM_nD_1Att_{}LSTM_nD.csv".format(A+1, B+1), 'a+') as predictcsv:
#     writer = csv.writer(predictcsv)
#     writer.writerow(["第n次", "test_loss"])

for A in range(A_layers):
    for B in range(B_layers):
        for neuron in neurons:
            total_loss = np.zeros((_epochs))
            total_val_loss = np.zeros((_epochs))
            total_test_loss = 0
            total_test_mae = 0
            if neuron == 256:
                _batch_size = 512
            for n in range(test_times):
                model = Sequential()
                model.add((LSTM(neuron,
                                input_shape=(_lookback, 3), 
                                return_sequences=True,
                                dropout=0.2,
                                recurrent_dropout=0.2
                                )))
                for aa in range(A):
                    model.add((LSTM(neuron, 
                                    return_sequences=True,
                                    dropout=0.2,
                                    recurrent_dropout=0.2
                                    )))
                model.add(attention())
                for bb in range(B):
                    model.add((LSTM(neuron, 
                                    return_sequences=True,
                                    dropout=0.2,
                                    recurrent_dropout=0.2
                                    )))
                model.add((LSTM(_delay, 
                                # activation=None,
                                return_sequences=False,
                                dropout=0.2,
                                recurrent_dropout=0.2
                                )))
                # -----------------------------------------------------
                model.summary()
                model.compile(optimizer=Adam(), 
                            loss='mse',
                            #   metrics=['mae']
                            )
                # checkpoint
                filepath="weights.best.hdf5"
                checkpoint = ModelCheckpoint(filepath, 
                                            monitor='val_loss', 
                                            verbose=1, 
                                            save_best_only=True,
                                            mode='min')
                callbacks_list = [checkpoint]

                history = model.fit(x_train, y_train,
                                    epochs=_epochs,
                                    batch_size=_batch_size,
                                    callbacks=callbacks_list,
                                    validation_split=0.3,
                                    verbose=1)

                model.load_weights("weights.best.hdf5")
                test_results = model.evaluate(x_test, y_test, batch_size=_batch_size)
                print("test_results:{}".format(test_results))
                total_test_loss += test_results

                predictions = model.predict(x_test, verbose=1, batch_size=_batch_size)
                print(predictions.shape)

                example_history = x_test[1069, :, 0].reshape(-1, 1)
                example_true_future = y_test[1069, :].reshape(-1, 1)
                x1 = np.arange(1, _lookback+1, 1)
                x2 = np.arange(_lookback+1, _lookback+_delay+1, 1)
                # ---------------------------------------------------------------------------------------------------------------------------------
                plt.figure()
                plt.plot(x1, example_history*(data_max[0]-data_min[0])+data_min[0], '-', color = 'r', ms=0.6, label = "history")
                plt.plot(x2, example_true_future*(data_max[0]-data_min[0])+data_min[0], 'o-', color = 'fuchsia', ms=0.6, label = "true_future")
                plt.plot(x2, predictions[1069, :]*(data_max[0]-data_min[0])+data_min[0], 's-', color = 'b', ms=0.6, label = "predict_future")    
                plt.title("one_of_test_results")
                plt.xlabel("time")
                plt.ylabel("Relative Humidity(%)")
                plt.legend()
                plt.savefig("{}th_{}neurons_test_result_{}LSTM_nD_1Att_{}LSTM_nD_1.png".format(n+1, neuron, A+1, B+1))
                # ---------------------------------------------------------------------------------------------------------------------------------
                example_history = x_test[2222, :, 0].reshape(-1, 1)
                example_true_future = y_test[2222, :].reshape(-1, 1)
                plt.figure()
                plt.plot(x1, example_history*(data_max[0]-data_min[0])+data_min[0], '-', color = 'r', ms=0.6, label = "history")
                plt.plot(x2, example_true_future*(data_max[0]-data_min[0])+data_min[0], 'o-', color = 'fuchsia', ms=0.6, label = "true_future")
                plt.plot(x2, predictions[2222, :]*(data_max[0]-data_min[0])+data_min[0], 's-', color = 'b', ms=0.6, label = "predict_future")    
                plt.title("one_of_test_results")
                plt.xlabel("time")
                plt.ylabel("Relative Humidity(%)")
                plt.legend()
                plt.savefig("{}th_{}neurons_test_result_{}LSTM_nD_1Att_{}LSTM_nD_2.png".format(n+1, neuron, A+1, B+1))
                # ---------------------------------------------------------------------------------------------------------------------------------
                example_history = x_test[69, :, 0].reshape(-1, 1)
                example_true_future = y_test[69, :].reshape(-1, 1)
                plt.figure()
                plt.plot(x1, example_history*(data_max[0]-data_min[0])+data_min[0], '-', color = 'r', ms=0.6, label = "history")
                plt.plot(x2, example_true_future*(data_max[0]-data_min[0])+data_min[0], 'o-', color = 'fuchsia', ms=0.6, label = "true_future")
                plt.plot(x2, predictions[69, :]*(data_max[0]-data_min[0])+data_min[0], 's-', color = 'b', ms=0.6, label = "predict_future")    
                plt.title("one_of_test_results")
                plt.xlabel("time")
                plt.ylabel("Relative Humidity(%)")
                plt.legend()
                plt.savefig("{}th_{}neurons_test_result_{}LSTM_nD_1Att_{}LSTM_nD_3.png".format(n+1, neuron, A+1, B+1))
                
                # print(model.layers[1].weights)
                # print("haha1")
                # print(model.layers[1].bias.numpy())
                # print("haha2")
                # print(model.layers[1].bias_initializer)
                # print("haha3")

                # Desired variable is called "attention_1/att_weight:0".
                # var = [v for v in tf.trainable_variables() if v.name == "attention_1/att_weight:0"]
                # print(type(var))
                # print(var)
                # sess = tf.Session()
                # sess.run(var)

                with open("{}LSTM_nD_1Att_{}LSTM_nD.csv".format(A+1, B+1), 'a+') as predictcsv:
                    writer = csv.writer(predictcsv)
                    # writer.writerow(["第n次", "test_loss"])
                    writer.writerow(["{}, {}".format(n+1, neuron), test_results])
                
                total_loss += np.array(history.history["loss"])
                total_val_loss += np.array(history.history["val_loss"])

            mean_test_loss = total_test_loss/test_times
            with open("{}LSTM_nD_1Att_{}LSTM_nD.csv".format(A+1, B+1), 'a+') as predictcsv:
                writer = csv.writer(predictcsv)
                # writer.writerow(["第n次", "test_loss", "test_mae"])
                writer.writerow(["mean,{}".format(neuron), mean_test_loss])
            epochs = range(1, len(total_loss)+1)
            mean_loss = total_loss/test_times
            mean_val_loss = total_val_loss/test_times
            plt.figure()
            plt.plot(epochs, mean_loss, 's-', color='b', ms=0.5, label="Training loss")
            plt.plot(epochs, mean_val_loss, 'o-', color='r', ms=0.5, label="Validation loss")
            plt.title("Training and validation loss (test {} time)".format(test_times))
            plt.xlabel("epochs")
            plt.ylabel("Mean Squared Error(MSE)")
            plt.legend()
            plt.savefig("{}_{}LSTM_nD_1Att_{}LSTM_nD_Mean_of_{}_test_loss.png".format(neuron, A+1, B+1, n+1))
