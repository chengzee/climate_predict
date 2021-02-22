import numpy as np
import pandas as pd
import csv 
import time
for number in range(6):
    # 讀取感測器資料
    rawdata = pd.read_csv("sensorNode9{}.csv".format(number+1)) 
    # 轉為矩陣
    rawdata_array = np.array(rawdata)

    total_LossNumber = 0
    # 另存名為 paddeddata_array 作為線性插植後的感測器資料
    paddeddata_array = rawdata_array
    for n in range(len(paddeddata_array)):
        paddeddata_array[n, 0]  = time.mktime(time.strptime(rawdata_array[n, 0], "%Y-%m-%d %H:%M:%S"))
    # 對感測器資料進行線性插植
    for n in range(len(rawdata_array)):
        if n > 0:
            # 計算時間差 (轉換為 unix timestamp)
            fronttime = rawdata_array[n-1, 0]
            latertime = rawdata_array[n, 0]
            diff_sec = latertime-fronttime
            # 差幾筆，從編號看
            LossNumber = int(rawdata_array[n:n+1, 4]-rawdata_array[n-1:n, 4])-1
            # 表示有漏接訊號，感測器並未重啟，將作 padding (此處指「線性插植」)
            if LossNumber > 0: 
                time_interval = diff_sec/(LossNumber+1)
                LossValue = (rawdata_array[n:n+1, 1:4]-rawdata_array[n-1:n, 1:4])/(LossNumber+1)
                padding_array = np.zeros((LossNumber, 5))
                for v in range(LossNumber):
                    padding_array[v:v+1, 0:1] = rawdata_array[n-1, 0] + time_interval*(v+1)
                    padding_array[v:v+1, 1:4] = rawdata_array[n-1:n, 1:4] + LossValue*(v+1)
                    # 若遺失數據超過二小時，會另行註記，之後將可能會不採用有包含該範圍的數據
                    if LossNumber < 24:
                        padding_array[v:v+1, 4:5] = int(rawdata_array[n-1:n, 4:5])+(v+1)
                    if LossNumber > 24:
                        padding_array[v:v+1, 4:5] = 0
                paddeddata_array = np.vstack((np.vstack((paddeddata_array[:n+total_LossNumber], padding_array)), paddeddata_array[n+total_LossNumber:]))
                total_LossNumber += LossNumber
            # 表示有重新開機，主因是沒電更換電池，通常應僅花費數十分鐘內，故將作padding
            if LossNumber < 0:
                # print("diff_sec:{}".format(diff_sec))
                # 300 (sec/step)
                diff_step = diff_sec/300 
                print("diff_step:{}".format(diff_step))
                if diff_step >= 2:
                    LossNumber = int(diff_step)-1
                    time_interval = diff_sec/(LossNumber+1)
                    LossValue = (rawdata_array[n:n+1, 1:4]-rawdata_array[n-1:n, 1:4])/(LossNumber+1)
                    # 如果二小時的關機後再重啟，會另行註記，之後將可能會不採用有包含該範圍的數據
                    if diff_step > 24:
                        padding_array = np.zeros((LossNumber, 5))
                    if diff_step <= 24:
                        padding_array = np.ones((LossNumber, 5))
                    for v in range(LossNumber):
                        padding_array[v:v+1, 0:1] = rawdata_array[n-1, 0] + time_interval*(v+1)
                        padding_array[v:v+1, 1:4] = rawdata_array[n-1:n, 1:4] + LossValue*(v+1)
                    paddeddata_array = np.vstack((np.vstack((paddeddata_array[:n+total_LossNumber], padding_array)), paddeddata_array[n+total_LossNumber:]))    
                    total_LossNumber += LossNumber

    with open("paddeddata9{}.csv".format(number+1), 'w', newline='') as thecsvfile:
        writer = csv.writer(thecsvfile)
        writer.writerow(["timestamps", "humidity", "temperature", "lightness", "count"])
        for row in paddeddata_array[:]:
            writer.writerow(row)
    with open("addfeature9{}.csv".format(number+1), 'w', newline='') as addfeaturecsvfile:
        writer = csv.writer(addfeaturecsvfile)
        writer.writerow(["timestamps", "humidity", "temperature", "12HrsHighTemp", "12HrsHighHumid"])
        for n in range(len(paddeddata_array)):
            Tempboolean = 0
            Humidboolean = 0
            if np.count_nonzero(paddeddata_array[n-144:n, 2]>29)/144 > 0.8:
                Tempboolean = 1
            if np.count_nonzero(paddeddata_array[n-144:n, 1]>75)/144 > 0.5:
                Humidboolean = 1
            writer.writerow([paddeddata_array[n][0], paddeddata_array[n][1], paddeddata_array[n][2], Tempboolean, Humidboolean])