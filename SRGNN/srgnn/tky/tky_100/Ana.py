import numpy as np


data_train = np.load('ERR_BEST_train_label_tky_100.npy')
data = np.load('ERR_BEST_train_tky_100.npy')
print(data_train.shape)
print(data[0])