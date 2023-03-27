from sklearn.neighbors import KNeighborsClassifier
import numpy as np
knn = KNeighborsClassifier(n_neighbors=3)
x_train = np.load("../srgnn/tky/tky_100/ERR_BEST_train_tky_100.npy")
y_train = np.load("../srgnn/tky/tky_100/ERR_BEST_train_label_tky_100.npy")
x_test = np.load("../srgnn/tky/tky_100/ERR_BEST_test_tky_100.npy")
y_test = np.load("../srgnn/tky/tky_100/ERR_BEST_test_label_tky_100.npy")
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train))
knn.predict(x_test)
print(knn.score(x_test,y_test))
