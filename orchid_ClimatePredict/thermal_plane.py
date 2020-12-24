import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
# width, length = (3, 4)
width, length = (3, 3)
humid= np.zeros((width, length))
temp = np.zeros((width, length))
for w in range(width):
    for l in range(length):
        try:
            rawdata = pd.read_csv("sensorNode{}.csv".format(3*l+w+1)) # 讀資料進來
            rawdata_array = np.array(rawdata) # 轉為矩陣
            # humid[w, l] =  3*l+w+1
            humid[w, l] = rawdata_array[-1, 1]
            temp[w, l] = rawdata_array[-1, 2]
        except FileNotFoundError:
            print("FileNotFoundError")
            # humid[pose] = 0
            # temp[pose] = 0
        except :
            print("Unexpected Error")
# 在opencv中是(x, y)
# dst_shape = (39, 28)
dst_shape = (2600, 2800)
humid_interpolation = cv2.resize(humid, dst_shape)
temp_interpolation = cv2.resize(temp, dst_shape)
print(humid_interpolation)
plt.figure("Humid")
plt.title("Humid", fontsize=18)
plt.imshow(humid, cmap="jet", origin="lower")
plt.colorbar()
plt.figure("Humid_interpolation")
plt.title("Humid Distribution", fontsize=18)
plt.imshow(humid_interpolation, cmap="jet", origin="lower")
plt.colorbar()
# ------------------------------------------------------------
plt.figure("temp")
plt.title("temp", fontsize=18)
plt.imshow(temp, cmap="jet", origin="lower")
plt.colorbar()
plt.figure("temp_interpolation")
plt.title("Temp Distribution", fontsize=18)
plt.imshow(temp_interpolation, cmap="jet", origin="lower")
plt.colorbar()
plt.show()
