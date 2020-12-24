import numpy as np
import pandas as pd
import csv 
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

# %env CUDA_VISIBLE_DEVICES=0

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 10000)])

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
neurons = [64, 128, 256, 512]
# neurons = [64, 128, 256]
# neurons = [256]
test_times = 10
_epochs = 100
_batch_size = 64
A_layers = 3
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

# with open("{}LSTM.csv".format(A+2), 'a+') as predictcsv:
#     writer = csv.writer(predictcsv)
#     writer.writerow(["第n次", "test_loss"])

for A in range(A_layers):
# for A in range(1, 3):
    for neuron in neurons:
        total_loss = np.zeros((_epochs))
        total_val_loss = np.zeros((_epochs))
        total_test_loss = 0
        total_test_mae = 0
        # if neuron == 256 or neuron == 512:
        #     _batch_size = 256
        # if A == 0 or A == 1:
        #     _epochs = 60
        for n in range(test_times):
            model = tf.keras.models.Sequential()
            model.add((CuDNNLSTM(neuron,
                       input_shape=(_lookback, 3), 
                       return_sequences=True, 
                    #    dropout=0.2,
                       # recurrent_dropout=0.2
                    #    activation='tanh',
                    #    recurrent_activation='sigmoid',
                    #    recurrent_dropout=0,
                       )))
            for aa in range(A):
                model.add((CuDNNLSTM(neuron, 
                                     return_sequences=True, 
                                    #  dropout=0.2,
                                     # recurrent_dropout=0.2
                                    #  activation='tanh',
                                    #  recurrent_activation='sigmoid',
                                    #  recurrent_dropout=0,
                                     )))
            model.add((CuDNNLSTM(_delay, 
                                 return_sequences=False
                                 )))
            # -----------------------------------------------------
            model.summary()
            model.compile(optimizer='adam', 
                          loss='mse',
                          #   metrics=['mae']
                          )
            # checkpoint
            filepath="weights.best.hdf5"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
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
            plt.savefig("{}th_{}neurons_test_result_{}LSTM_1.png".format(n+1, neuron, A+2))
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
            plt.savefig("{}th_{}neurons_test_result_{}LSTM_2.png".format(n+1, neuron, A+2))
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
            plt.savefig("{}th_{}neurons_test_result_{}LSTM_3.png".format(n+1, neuron, A+2))

            with open("{}LSTM.csv".format(A+2), 'a+') as predictcsv:
                writer = csv.writer(predictcsv)
                # writer.writerow(["第n次", "test_loss"])
                writer.writerow(["{}, {}".format(n+1, neuron), test_results])
            
            total_loss += np.array(history.history["loss"])
            total_val_loss += np.array(history.history["val_loss"])

        mean_test_loss = total_test_loss/test_times
        with open("{}LSTM.csv".format(A+2), 'a+') as predictcsv:
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
        plt.savefig("{}_{}LSTM_Mean_of_{}_test_loss.png".format(neuron, A+2, n+1))

# ---------------------------------------------------------------------------------------------------------
# model = tf.keras.Sequential()
# model.add((tf.keras.layers.LSTM(64,
#                 input_shape=(_lookback, 3), 
#                 # batch_input_shape=(_batch_size, _lookback, 3),
#                 return_sequences=True, 
#                 dropout=0.2,
#                 # recurrent_dropout=0.2
#                 activation='tanh',
#                 recurrent_activation='sigmoid',
#                 recurrent_dropout=0,
#                 )))
# # for aa in range(A):
# #     model.add((LSTM(neuron, 
# #                     return_sequences=True, 
# #                     dropout=0.2,
# #                     recurrent_dropout=0.2
# #                     )))
# model.add((tf.keras.layers.LSTM(_delay, 
#                 return_sequences=False,
#                 activation='tanh',
#                 recurrent_activation='sigmoid',
#                 recurrent_dropout=0,
#                 )))
# # -----------------------------------------------------
# model.summary()
# model.compile(optimizer=tf.keras.optimizers.Adam(), 
#             loss='mse',
#             #   metrics=['mae']
#             )


# history = model.fit(x_train, y_train,
#                     epochs=_epochs,
#                     batch_size=512,
#                     # callbacks=callbacks_list,
#                     validation_split=0.3,
#                     verbose=1)