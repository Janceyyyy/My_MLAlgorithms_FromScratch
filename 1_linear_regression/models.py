import random
import numpy as np


def squared_error(predictions, Y):
    '''
    Computes sum squared loss (the L2 loss) between true values, Y, and predictions.
    @params:
        Y: A 1D Numpy array with real values (float64)
        predictions: A 1D Numpy array of the same size of Y
    @return:
        sum squared loss (the L2 loss) using predictions for Y.
    '''

    return np.sum((Y - predictions)**2)


class LinearRegression:
    '''
    LinearRegression model that minimizes squared error using matrix inversion.
    '''
    def __init__(self, n_features):
        '''
        @attrs:
            n_features: the number of features in the regression problem
            weights: The weights of the linear regression model.
        '''
        self.n_features = n_features + 1  # An extra feature added for the bias value
        self.weights = np.zeros(n_features + 1)

    def train(self, X, Y):
        '''
        Trains the LinearRegression model by finding the optimal set of weights
        using matrix inversion.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            the weights of the regression model
        '''
        # [TODO]
        
        X_dot_X = np.dot(X.T, X)
        X_dot_X_inv = np.linalg.inv(X_dot_X)
        X_inv_dot_X_T = np.dot(X_dot_X_inv, X.T)
        self.weights = np.dot(X_inv_dot_X_T, Y)
        return self.weights
    
    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        X = np.dot(X,self.weights) # adding the bias feature
        return X 


    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            A float number which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return squared_error(predictions, Y)


    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).
        MSE = Total squared error/# of examples
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            A float number which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]
