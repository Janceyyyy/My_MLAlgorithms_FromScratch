

import numpy as np
import random
from sklearn.model_selection import train_test_split
from models import LinearRegression

from sklearn.model_selection import train_test_split

WINE_FILE_PATH = 'data/wine.txt'

def import_wine(filepath, test_size=0.2):
    '''
        Helper function to import the wine dataset
        @param:
            filepath: path to wine.txt
            test_size: the fraction of the dataset set aside for testing
        @return:
            X_train: training data inputs
            Y_train: training data values
            X_test: testing data inputs
            Y_test: testing data values
    '''

    # Load in the dataset
    data = np.loadtxt(filepath, skiprows=1)
    X, Y = data[:, 1:], data[:, 0]

    # Normalize the inputs
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def test_linreg():
    '''
        Helper function that tests LinearRegression.
    '''

    X_train, X_test, Y_train, Y_test = import_wine(WINE_FILE_PATH)

    num_features = X_train.shape[1]

    # Padding the inputs with a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    #### Matrix Inversion ######
    print('---- LINEAR REGRESSION w/ Matrix Inversion ---')
    solver_model = LinearRegression(num_features)
    solver_model.train(X_train_b, Y_train)
    print('Average Training Loss:', solver_model.average_loss(X_train_b, Y_train))
    print('Average Testing Loss:', solver_model.average_loss(X_test_b, Y_test))



def main():
    random.seed(0)
    np.random.seed(0)
    test_linreg()

if __name__ == "__main__":
    main()
