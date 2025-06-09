import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def fit(self, X, y):
        no_of_samples, no_of_features = X.shape
        self.weights = np.zeros(no_of_features)
        self.bias = 0
        y_reshaped = y.reshape(no_of_samples, 1)
        for i in range (self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1/no_of_samples) * np.dot(X.T, (y_predicted - y_reshaped))
            db = (1/no_of_samples) * np.sum(y_predicted - y_reshaped)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)