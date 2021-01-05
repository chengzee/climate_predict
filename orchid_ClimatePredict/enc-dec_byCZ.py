import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import os
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_file_path = './Enc-Dec_byCZ'
if not os.path.isdir(save_file_path):
    os.mkdir(save_file_path)

paddeddata_csv = pd.read_csv("paddeddata1.csv")  # 讀資料進來
paddeddata_array = np.array(paddeddata_csv)  # 轉為矩陣
# -------------------------------------------------------------------------------------------------------------------
# 最小最大值正規化 [0, 1]
data_min = np.min(paddeddata_array[:, 1:4], axis=0)
data_max = np.max(paddeddata_array[:, 1:4], axis=0)
# data_mean = np.mean(paddeddata_array[:, 1:4], axis=0)
print("data_min:{}".format(data_min))
print("data_max:{}".format(data_max))
# # print("data_mean:{}".format(data_mean))
paddeddata_array_norm = paddeddata_array
paddeddata_array_norm[:, 1:4] = (paddeddata_array[:, 1:4]-data_min)\
    / (data_max-data_min)
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
source_dim = 3
predict_dim = 1
test_times = 10
BATCH_SIZE = 256
_epochs = 150
A_layers = 5

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
        GenDataset(inputdata=paddeddata_array_norm,
                   starttime=the_first_nonzero,
                   lasttime=the_last_nonzero,
                   lookback=_lookback,
                   delay=_delay,
                   samp_list=sample_list,
                   targ_list=target_list)
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

for A in range(A_layers):
    for neuron in neurons:
        total_loss = np.zeros((_epochs))
        total_val_loss = np.zeros((_epochs))
        total_test_mse = 0
        total_test_mae = 0

        with open(save_file_path+"/{}LSTM_Enc-Dec.csv".format(A+1), 'a+') as predictcsv:
            writer = csv.writer(predictcsv)
            writer.writerow(["第n次", "test_mse", "test_mae"])

        for n in range(test_times):

            # Sequential
            # input_seq = tf.keras.Input(_lookback, source_dim)
            model = tf.keras.Sequential()
            model.add(tf.keras.Input(shape=(_lookback, source_dim)))
            for aa in range(A):
                model.add(tf.keras.layers.LSTM(neuron, return_sequences=True))
            model.add(tf.keras.layers.LSTM(neuron, return_sequences=False))

            # # RepeatVector參考來源如下
            # # https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
            model.add(tf.keras.layers.RepeatVector(_delay))
            # # ----------------------------------------------------------------------------------------------------
            # # 自己的想法...
            # model.add(tf.keras.layers.Dense(_delay))
            # model.add(tf.keras.layers.Reshape((_delay, 1)))
            # # ----------------------------------------------------------------------------------------------------

            model.add(tf.keras.layers.LSTM(neuron, return_sequences=True))
            for aa in range(A):
                model.add(tf.keras.layers.LSTM(neuron, return_sequences=True))
            model.add(tf.keras.layers.Dense(predict_dim))

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            model.summary()
            # checkpoint
            filepath = save_file_path + "/weights.best.h5"
            # filepath = save_file_path + "/training_checkpoints.ckpt"

            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                            monitor='val_loss',
                                                            verbose=1,
                                                            save_best_only=True,
                                                            # save_weights_only=True,
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
            fig1 = plt.figure()
            plt.plot(x1, example_history*(data_max[0]-data_min[0])+data_min[0], '-', color='r', ms=0.6, label="history")
            plt.plot(x2, example_true_future*(data_max[0]-data_min[0])+data_min[0], 'o-', color='fuchsia', ms=0.6, label="true_future")
            plt.plot(x2, predictions[1069, :]*(data_max[0]-data_min[0])+data_min[0], 's-', color='b', ms=0.6, label="predict_future")
            plt.title("Encoder-Decoder({}neurons_{}_layers_LSTM)".format(neuron, A+1))
            plt.xlabel("time")
            plt.ylabel("Relative Humidity(%)")
            plt.legend()
            plt.savefig(save_file_path+"/{}th_{}neurons_{}ENC-DEC_288-72_1.png".format(n+1, neuron, A+1))
            plt.close(fig1)
            # ---------------------------------------------------------------------------------------------------------------------------------
            example_history = x_test[2222, :, 0].reshape(-1, 1)
            example_true_future = y_test[2222, :].reshape(-1, 1)
            fig2 = plt.figure()
            plt.plot(x1, example_history*(data_max[0]-data_min[0])+data_min[0], '-', color='r', ms=0.6, label="history")
            plt.plot(x2, example_true_future*(data_max[0]-data_min[0])+data_min[0], 'o-', color='fuchsia', ms=0.6, label="true_future")
            plt.plot(x2, predictions[2222, :]*(data_max[0]-data_min[0])+data_min[0], 's-', color='b', ms=0.6, label="predict_future")
            plt.title("Encoder-Decoder({}neurons_{}_layers_LSTM)".format(neuron, A+1))
            plt.xlabel("time")
            plt.ylabel("Relative Humidity(%)")
            plt.legend()
            plt.savefig(save_file_path+"/{}th_{}neurons_{}ENC-DEC_288-72_2.png".format(n+1, neuron, A+1))
            plt.close(fig2)
            # ---------------------------------------------------------------------------------------------------------------------------------
            example_history = x_test[69, :, 0].reshape(-1, 1)
            example_true_future = y_test[69, :].reshape(-1, 1)
            fig3 = plt.figure()
            plt.plot(x1, example_history*(data_max[0]-data_min[0])+data_min[0], '-', color='r', ms=0.6, label="history")
            plt.plot(x2, example_true_future*(data_max[0]-data_min[0])+data_min[0], 'o-', color='fuchsia', ms=0.6, label="true_future")
            plt.plot(x2, predictions[69, :]*(data_max[0]-data_min[0])+data_min[0], 's-', color='b', ms=0.6, label="predict_future")
            plt.title("Encoder-Decoder({}neurons_{}_layers_LSTM)".format(neuron, A+1))
            plt.xlabel("time")
            plt.ylabel("Relative Humidity(%)")
            plt.legend()
            plt.savefig(save_file_path+"/{}th_{}neurons_{}ENC-DEC_288-72_3.png".format(n+1, neuron, A+1))
            plt.close(fig3)

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
        figMSE = plt.figure()
        plt.plot(epochs, mean_loss, 's-', color='b', ms=0.5, label="Training loss")
        plt.plot(epochs, mean_val_loss, 'o-', color='r', ms=0.5, label="Validation loss")
        plt.title("Training and validation loss (test {} time)".format(test_times))
        plt.xlabel("epochs")
        plt.ylabel("Mean Squared Error(MSE)")
        plt.legend()
        plt.savefig(save_file_path+"/{}_{}Enc-Dec_Mean_of_10time_test_MSE.png".format(neuron, A+1))
        plt.close(figMSE)
        rmse_mean_loss = mean_loss**0.5
        rmse_mean_val_loss = mean_val_loss**0.5
        figRMSE = plt.figure()
        plt.plot(epochs, rmse_mean_loss, 's-', color='b', ms=0.5, label="Training loss")
        plt.plot(epochs, rmse_mean_val_loss, 'o-', color='r', ms=0.5, label="Validation loss")
        plt.title("Training and validation loss (test {} time)".format(test_times))
        plt.xlabel("epochs")
        plt.ylabel("Root Mean Squared Error(RMSE)")
        plt.legend()
        plt.savefig(save_file_path+"/{}_{}Enc-Dec_Mean_of_10time_test_RMSE.png".format(neuron, A+1))
        plt.close(figRMSE)
