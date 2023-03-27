from sklearn.naive_bayes import GaussianNB
import numpy as np

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# x_train = np.load("../SR-GNN/pytorch_sr_gnn/data/jazz/MMR_BEST_train_jazz_400.npy")
# y_train = np.load("../SR-GNN/pytorch_sr_gnn/data/jazz/MMR_BEST_train_label_jazz_400.npy")
# x_test = np.load("../SR-GNN/pytorch_sr_gnn/data/jazz/MMR_BEST_test_jazz_400.npy")
# y_test = np.load("../SR-GNN/pytorch_sr_gnn/data/jazz/MMR_BEST_test_label_jazz_400.npy")
x_train = np.load("../srgnn/tky/tky_100/ERR_BEST_train_tky_100.npy")
y_train = np.load("../srgnn/tky/tky_100/ERR_BEST_train_label_tky_100.npy")
x_test = np.load("../srgnn/tky/tky_100/ERR_BEST_test_tky_100.npy")
y_test = np.load("../srgnn/tky/tky_100/ERR_BEST_test_label_tky_100.npy")
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.fit_transform(x_test)
mlt = GaussianNB()
mlt.fit(X_train_scaled, y_train)

print("准确率为：", mlt.score(X_train_scaled, y_train))