import numpy as np
import pandas as pd
import csv 
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import os
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_file_path = './Enc-Dec_byCZ'
if not os.path.isdir(save_file_path):
    os.mkdir(save_file_path)

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
count = 1
the_first_nonzero = 0
the_last_nonzero = 0
n = 0
_lookback = 288
_delay = 12*6
sample_list = []
target_list = []
train_size = 0.7
neurons = [64, 128, 256]
source_dim = 3
predict_dim = 1
test_times = 10
_epochs = 100
BUFFER_SIZE = 65535
BATCH_SIZE = 256
A_layers = 5

class Encode(tf.keras.Model):
    def __init__(self, enc_units, _layers):
        super(Encode, self).__init__()
        self.enc_units = enc_units
        self.layer = _layers
        # self.lstm_returndseqTrue = tf.keras.layers.LSTM(self.enc_units,
        #                                             return_sequences=True)
        self.lstm_returnTrue = tf.keras.layers.LSTM(self.enc_units,
                                                    return_sequences=True,
                                                    return_state=True)
        self.lstm_returnFalse = tf.keras.layers.LSTM(self.enc_units)
        # self.last_lstm = tf.keras.layers.LSTM(self.enc_units,
        #                                       return_sequences=True,
        #                                       return_state=True)
    def call(self, x):
        stack_hidden_state = x
        # stack_hidden_state = self.lstm(x)
        stack_hidden_state, last_hidden_state, last_cell_state = self.lstm_returnTrue(stack_hidden_state)
        for n in range(self.layer):
            stack_hidden_state, last_hidden_state, last_cell_state = self.lstm_returnTrue(stack_hidden_state, initial_state=[last_hidden_state, last_cell_state])
        last_hidden_state = self.lstm_returnFalse(stack_hidden_state, initial_state=[last_hidden_state, last_cell_state])
        # Encoder_output = last_stack_hidden_state
        # last_stack_hidden_state, last_hidden_state, last_cell_state = self.last_lstm(stack_hidden_state)
        # return last_stack_hidden_state, last_hidden_state, last_cell_state 
        return last_hidden_state 
        
class Decode(tf.keras.Model):
    def __init__(self, dec_units, _layers):
        super(Decode, self).__init__()
        self.dec_units = dec_units
        self.layer = _layers
        self.lstm_returnTrue = tf.keras.layers.LSTM(self.dec_units,
                                                    return_state=True, 
                                                    return_sequences=True)
        # self.lstm_returnFalse = tf.keras.layers.LSTM(self.dec_units,
        #                                              return_sequences=False)
    def call(self, x):
        stack_hidden_state = x
        stack_hidden_state, last_hidden_state, last_cell_state = self.lstm_returnTrue(stack_hidden_state)
        for n in range((self.layer)):
            stack_hidden_state, last_hidden_state, last_cell_state = self.lstm_returnTrue(stack_hidden_state, initial_state=[last_hidden_state, last_cell_state])
        stack_hidden_state, last_hidden_state, last_cell_state = self.lstm_returnTrue(stack_hidden_state, initial_state=[last_hidden_state, last_cell_state])
        return stack_hidden_state 


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

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# # print(train_dataset.element_spec)
# # print(test_dataset.element_spec)

# train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
# test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

for A in range(1, 5):
    for neuron in neurons:
        total_loss = np.zeros((_epochs))
        total_val_loss = np.zeros((_epochs))
        total_test_mse = 0
        total_test_mae  = 0
        
        with open(save_file_path+"/{}LSTM_Enc-Dec.csv".format(A+1), 'a+') as predictcsv:
            writer = csv.writer(predictcsv)
            writer.writerow(["第n次", "test_mse", "test_mae"])

        for n in range(test_times):
            
            # # Functional----------------------------------------------------------------------
            # encoder_input = tf.keras.Input(shape=(_lookback, source_dim))

            # # for aa in range(A):
            # #     encoder_input = tf.keras.layers.LSTM(neuron, 
            # #                                           return_sequences=True, 
            # #                                         #   dropout=0.2,
            # #                                         #   recurrent_dropout=0,
            # #                                           )(encoder_input)
            # #encoder_input = tf.keras.layers.LSTM(neuron, 
            # #                                      return_sequences=False, 
            # #                                    #   dropout=0.2,
            # #                                    #   recurrent_dropout=0,
            # #                                      )(encoder_input)

            # encoder = Encode(neuron, A)
            # encoder_output = encoder(encoder_input)

            
            # # RepeatVector參考來源如下
            # # https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
            # # decoder_input = tf.keras.layers.RepeatVector(_delay)(encoder_output)
            # # ----------------------------------------------------------------------------------------------------
            # # 自己的想法...
            # decoder_input = tf.keras.layers.Dense(_delay)(encoder_output)
            # decoder_input = tf.reshape(decoder_input, [-1, _delay, 1])
            # # ----------------------------------------------------------------------------------------------------
            # print(tf.shape(decoder_input))

            # decoder = Decode(neuron, A)
            # decoder_output = decoder(decoder_input)

            # # for aa in range(A):
            # #     decoder_input = tf.keras.layers.LSTM(neuron,
            # #                                          return_sequences=True,
            # #                                         #  recurrent_dropout=0
            # #                                          )(decoder_input)
                                                     
            # # decoder_output = tf.keras.layers.LSTM(neuron,
            # #                                       return_sequences=True,
            # #                                     #   recurrent_dropout=0, 
            # #                                       )(decoder_input)

            # # predict_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(predict_dim))(decoder_output)
            # predict_output = tf.keras.layers.Dense(predict_dim)(decoder_output)

            # predict_output = tf.reshape(predict_output, [-1, _delay])
            # -----------------------------------------------------
            # Sequential
            # input_seq = tf.keras.Input(_lookback, source_dim)
            model = tf.keras.Sequential()
            model.add(tf.keras.Input(shape=(_lookback, source_dim)))
            for aa in range(A):
                model.add(tf.keras.layers.LSTM(neuron, return_sequences=True))
            model.add(tf.keras.layers.LSTM(neuron, return_sequences=False))
            model.add(tf.keras.layers.Dense(_delay))
            model.add(tf.keras.layers.Reshape((_delay, 1)))
            model.add(tf.keras.layers.LSTM(neuron, return_sequences=True))
            for aa in range(A):
                model.add(tf.keras.layers.LSTM(neuron, return_sequences=True))
            model.add(tf.keras.layers.Dense(predict_dim))

            # -------------------------------------------------------
            # encoder_input = tf.keras.Input(shape=(_lookback, source_dim))
            # for aa in range(A):
            #     next_encoder_input = tf.keras.layers.LSTM(neuron, 
            #                                          return_sequences=True, 
            #                                         #   dropout=0.2,
            #                                         #   recurrent_dropout=0,
            #                                          )(encoder_input)
            #     encoder_input = next_encoder_input

            # encoder_output = tf.keras.layers.LSTM(neuron, 
            #                                       return_sequences=False,
            #                                     #   dropout=0.2,
            #                                     #   recurrent_dropout=0
            #                                       )(next_encoder_input)

            # decoder_input = tf.keras.layers.Dense(_delay)(encoder_output)
            # decoder_input = tf.reshape(decoder_input, [-1, _delay, 1])
            # for aa in range(A):
            #     next_decoder_input = tf.keras.layers.LSTM(neuron, 
            #                                          return_sequences=True, 
            #                                         #   dropout=0.2,
            #                                         #   recurrent_dropout=0,
            #                                          )(decoder_input)
            #     decoder_input = next_decoder_input
            
            # decoder_output = tf.keras.layers.LSTM(neuron, 
            #                                       return_sequences=True,
            #                                     #   dropout=0.2,
            #                                     #   recurrent_dropout=0
            #                                       )(next_decoder_input)
            
            # predict_output = tf.keras.layers.Dense(predict_dim)(decoder_output)

            # model = tf.keras.models.Model(inputs=encoder_input, outputs=predict_output)
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            model.summary()
            # checkpoint
            filepath = save_file_path + "/weights.best.h5"
            # filepath = save_file_path + "/training_checkpoints.ckpt"

            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, 
                                                            monitor='val_loss', 
                                                            verbose=1, 
                                                            save_best_only=True,
                                                            save_weights_only=True, 
                                                            mode='min')
            callbacks_list = [checkpoint]

            history = model.fit(x_train, y_train,
                                epochs=_epochs,
                                batch_size=BATCH_SIZE,
                                callbacks=callbacks_list,
                                validation_split=0.3,
                                verbose=1)

            model.load_weights(filepath)
            test_mse, test_mae = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
            print("test_results:{}, {}".format(test_mse, test_mae))
            total_test_mse += test_mse
            total_test_mae += test_mae

            predictions = model.predict(x_test, verbose=1, batch_size=BATCH_SIZE)
            print("predictions:{}".format(predictions))

            x1 = np.arange(1, _lookback+1, 1)
            x2 = np.arange(_lookback+1, _lookback+_delay+1, 1)
            # ---------------------------------------------------------------------------------------------------------------------------------
            example_history = x_test[1069, :, 0].reshape(-1, 1)
            example_true_future = y_test[1069, :].reshape(-1, 1)            
            plt.figure()
            plt.plot(x1, example_history*(data_max[0]-data_min[0])+data_min[0], '-', color = 'r', ms=0.6, label = "history")
            plt.plot(x2, example_true_future*(data_max[0]-data_min[0])+data_min[0], 'o-', color = 'fuchsia', ms=0.6, label = "true_future")
            plt.plot(x2, predictions[1069, :]*(data_max[0]-data_min[0])+data_min[0], 's-', color = 'b', ms=0.6, label = "predict_future")    
            plt.title("Encoder-Decoder({}neurons_{}_layers_LSTM)".format(neuron, A+1))
            plt.xlabel("time")
            plt.ylabel("Relative Humidity(%)")
            plt.legend()
            plt.savefig(save_file_path+"/{}th_{}neurons_{}ENC-DEC_288-72_1.png".format(n+1, neuron, A+1))
            # ---------------------------------------------------------------------------------------------------------------------------------
            example_history = x_test[2222, :, 0].reshape(-1, 1)
            example_true_future = y_test[2222, :].reshape(-1, 1)
            plt.figure()
            plt.plot(x1, example_history*(data_max[0]-data_min[0])+data_min[0], '-', color = 'r', ms=0.6, label = "history")
            plt.plot(x2, example_true_future*(data_max[0]-data_min[0])+data_min[0], 'o-', color = 'fuchsia', ms=0.6, label = "true_future")
            plt.plot(x2, predictions[2222, :]*(data_max[0]-data_min[0])+data_min[0], 's-', color = 'b', ms=0.6, label = "predict_future")    
            plt.title("Encoder-Decoder({}neurons_{}_layers_LSTM)".format(neuron, A+1))
            plt.xlabel("time")
            plt.ylabel("Relative Humidity(%)")
            plt.legend()
            plt.savefig(save_file_path+"/{}th_{}neurons_{}ENC-DEC_288-72_2.png".format(n+1, neuron, A+1))
            # ---------------------------------------------------------------------------------------------------------------------------------
            example_history = x_test[69, :, 0].reshape(-1, 1)
            example_true_future = y_test[69, :].reshape(-1, 1)
            plt.figure()
            plt.plot(x1, example_history*(data_max[0]-data_min[0])+data_min[0], '-', color = 'r', ms=0.6, label = "history")
            plt.plot(x2, example_true_future*(data_max[0]-data_min[0])+data_min[0], 'o-', color = 'fuchsia', ms=0.6, label = "true_future")
            plt.plot(x2, predictions[69, :]*(data_max[0]-data_min[0])+data_min[0], 's-', color = 'b', ms=0.6, label = "predict_future")    
            plt.title("Encoder-Decoder({}neurons_{}_layers_LSTM)".format(neuron, A+1))
            plt.xlabel("time")
            plt.ylabel("Relative Humidity(%)")
            plt.legend()
            plt.savefig(save_file_path+"/{}th_{}neurons_{}ENC-DEC_288-72_3.png".format(n+1, neuron, A+1))

            with open(save_file_path+"/{}LSTM_Enc-Dec.csv".format(A+1), 'a+') as predictcsv:
                writer = csv.writer(predictcsv)
                # writer.writerow(["第n次", "test_mse", "test_mae"])
                writer.writerow(["{}, {}".format(n+1, neuron), test_mse, test_mae])
            
            total_loss += np.array(history.history["loss"])
            total_val_loss += np.array(history.history["val_loss"])

        mean_test_mse = total_test_mse/test_times
        mean_test_mae = total_test_mae/test_times
        with open(save_file_path+"/{}LSTM_Enc-Dec.csv".format(A+1), 'a+') as predictcsv:
            writer = csv.writer(predictcsv)
            # writer.writerow(["第n次", "test_loss", "test_mae"])
            writer.writerow(["mean,{}".format(neuron), mean_test_mse, mean_test_mae])
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
        plt.savefig(save_file_path+"/{}_{}Enc-Dec_Mean_of_10time_test_MSE.png".format(neuron, A+1))

        rmse_mean_loss = mean_loss**0.5
        rmse_mean_val_loss = mean_val_loss**0.5
        plt.figure()
        plt.plot(epochs, rmse_mean_loss, 's-', color='b', ms=0.5, label="Training loss")
        plt.plot(epochs, rmse_mean_val_loss, 'o-', color='r', ms=0.5, label="Validation loss")
        plt.title("Training and validation loss (test {} time)".format(test_times))
        plt.xlabel("epochs")
        plt.ylabel("Root Mean Squared Error(RMSE)")
        plt.legend()
        plt.savefig(save_file_path+"/{}_{}Enc-Dec_Mean_of_10time_test_RMSE.png".format(neuron, A+1))
