from sklearn.neighbors import KNeighborsClassifier
import numpy as np
knn = KNeighborsClassifier(n_neighbors=3)
x_train = np.load("../SR-GNN/pytorch_sr_gnn/data/hiv/MMR_BEST_train_hiv_200.npy")
y_train = np.load("../SR-GNN/pytorch_sr_gnn/data/hiv/MMR_BEST_train_label_hiv_200.npy")
x_test = np.load("../SR-GNN/pytorch_sr_gnn/data/hiv/MMR_BEST_test_hiv_200.npy")
y_test = np.load("../SR-GNN/pytorch_sr_gnn/data/hiv/MMR_BEST_test_label_hiv_200.npy")
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train))
knn.predict(x_test)
print(knn.score(x_test,y_test))
