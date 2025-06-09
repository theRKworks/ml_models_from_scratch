import numpy as np
from collections import Counter
class KNN:
    def __init__(self,k=3):
        self.k = k
    def fit(self,x,y):
        self.X_train = x
        self.Y_train = y

    def predict_point(self,x):
        distances = np.linalg.norm(self.X_train - x,axis = 1)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.Y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    def predict(self,x_test):
        predictions = [self.predict_point(x) for x in x_test]
        return np.array(predictions)
    
    # Toy dataset
X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 8]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[3, 3]])



# KNN model
knn = KNN(k=3)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

print("Prediction:", preds)  # Output should be 0 or 1 based on neighbors
