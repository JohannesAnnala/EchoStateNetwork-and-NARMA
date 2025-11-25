import numpy as np

#Ridge regression class that performs batch learning and sequantial learning
class RidgeRegression:

    def __init__(self, learning_rate, regularization_strength, iterations, sequential_learning=False):
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.iterations = iterations
        self.sequential_learning = sequential_learning

    def update_constants(self, learning_rate=None, regularization_strength=None):
        if learning_rate:
            self.learning_rate = learning_rate
        if regularization_strength:
            self.regularization_strength = regularization_strength

    def predict(self, X):
        return X @ self.W_out + self.bias_term

    def fit(self, X, Y):
        
        self.W_out = np.zeros(X.shape[1])
        self.bias_term = 0
               
        if self.sequential_learning:
            for _ in range(self.iterations):
                for x, y in zip(X,Y):
                    y_pred_ = self.predict(x)
                    W_out_gradient_ = - np.dot(x.T, (y - y_pred_)) + self.regularization_strength*self.W_out
                    bias_term_gradient_ = - np.sum(y - y_pred_)
                    self.W_out -= self.learning_rate*W_out_gradient_
                    self.bias_term -= self.learning_rate*bias_term_gradient_
        else:
            for _ in range(self.iterations):
                Y_pred_ = self.predict(X)
                W_out_gradient_ = - X.T @ (Y - Y_pred_) + self.regularization_strength*self.W_out
                bias_term_gradient_ = - np.sum(Y - Y_pred_)
                self.W_out = self.W_out - self.learning_rate*W_out_gradient_
                self.bias_term = self.bias_term - self.learning_rate*bias_term_gradient_


