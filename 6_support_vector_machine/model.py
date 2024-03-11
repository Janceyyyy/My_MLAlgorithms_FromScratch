'''
   This file contains functions to implement support vector machine (SVM) model
   to solve binary classification problems
   A Python library for solving quadratic problems, quadprog, is used as an optimizer
   instead of gradient-based methods
'''

import numpy as np
from qp import solve_QP

def linear_kernel(xi, xj):
    """
    Kernel Function, linear kernel (ie: regular dot product)
    @param xi: an input sample (1D np array)
    @param xj: an input sample (1D np array)
    @return: float64
    """
    #TODO
    return np.dot(xi,xj)


def rbf_kernel(xi, xj, gamma=0.1):
    """
    Kernel Function, radial basis function kernel
    @param xi: an input sample (1D np array)
    @param xj: an input sample (1D np array)
    @param gamma: parameter of the RBF kernel (scalar)
    @return: float64
    """
    # TODO
    return np.exp(-gamma * np.linalg.norm(xi - xj)**2)

def polynomial_kernel(xi, xj, c=2, d=2):
    """
    Kernel Function, polynomial kernel
    @param xi: an input sample (1D np array)
    @param xj: an input sample (1D np array)
    @param c: mean of the polynomial kernel (scalar)
    @param d: exponent of the polynomial (scalar)
    @return: float64
    """
    #TODO
    return (np.dot(xi,xj)+c) ** d


class SVM(object):

    def __init__(self, kernel_func=linear_kernel, lambda_param=.1):
        self.kernel_func = kernel_func
        self.lambda_param = lambda_param

    def train(self, inputs, labels):
        """
        train the model with the input data (inputs and labels),
        find the coefficients and constaints for the quadratic program and
        calculate the alphas
        @param inputs: inputs of data, a numpy array
        @param labels: labels of data, a numpy array
        @return: None
        """
        self.train_inputs = inputs
        self.train_labels = labels

        # constructing QP variables
        G = self._get_gram_matrix()
        Q, c = self._objective_function(G)
        A, b = self._inequality_constraint(G)


        # TODO: Uncomment the next line when you have implemented _get_gram_matrix(),
        #   _inequality_constraints() and _objective_function().
        self.alpha = solve_QP(Q, c, A, b)[:self.train_inputs.shape[0]]


    def _get_gram_matrix(self):
        """
        Generate the Gram matrix for the training data stored in self.train_inputs.
        Recall that element i, j of the matrix is K(x_i, x_j), where K is the
        kernel function.
        @return: the Gram matrix for the training data, a numpy array
        """

        # TODO 
        num_samples = self.train_inputs.shape[0]
        G = np.zeros((num_samples, num_samples))

        for i in range(num_samples):
            for j in range(num_samples):
                G[i, j] = self.kernel_func(self.train_inputs[i], self.train_inputs[j])
        return G

    def _objective_function(self, G):
        """
        Generate the coefficients on the variables in the objective function for the
        SVM quadratic program.
        Recall the objective function is:
        minimize (1/2)x^T Q x + c^T x
        @param G: the Gram matrix for the training data, a numpy array
        @return: two numpy arrays, Q and c which fully specify the objective function
        """

        # TODO
        Q = 2 * self.lambda_param * G
        m = self.train_inputs.shape[0]
        matrix = np.zeros((m, m))

        top_row = np.concatenate((Q, matrix), axis=1)
        bottom_row = np.concatenate((matrix, matrix), axis=1)
        Q = np.concatenate((top_row, bottom_row), axis=0)
        
        c = np.zeros(2 * m)
        c[m:] = np.ones(m) / m

        return Q, c

    def _inequality_constraint(self, G):
        """
        Generate the inequality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ax <= b.
        @param G: the Gram matrix for the training data, a numpy array
        @return: two numpy arrays, A and b which fully specify the constraints
        """
        m = self.train_inputs.shape[0]
        a1 = np.hstack((np.zeros((m, m)), -1*np.eye(m)))

        a2 = np.hstack((-np.diag(self.train_labels)@G, -1*np.eye(m)))
        
        A = np.vstack((a1, a2))
        
        zeros_array = np.zeros(m)
        negative_ones_array = -np.ones(m)
        b = np.hstack((zeros_array, negative_ones_array))

        return A, b


        



    def predict(self, inputs):
        """
        Generate predictions given input.
        @param input: 2D Numpy array. Each row is a vector for which we output a prediction.
        @return: A 1D numpy array of predictions.
        """


        predictions = []

        for i in inputs:
            sum_kernel = 0
            for j in range(self.train_inputs.shape[0]):
                sum_kernel += self.alpha[j] * self.kernel_func(self.train_inputs[j], i)
            predictions.append( np.sign(sum_kernel) or 1)

        return np.array(predictions)
    
    

    def accuracy(self, inputs, labels):
        """
        Calculate the accuracy of the classifer given inputs and their true labels.
        @param inputs: 2D Numpy array which we are testing calculating the accuracy of.
        @param labels: 1D Numpy array with the inputs corresponding true labels.
        @return: A float indicating the accuracy (between 0.0 and 1.0)
        """

        predictions = self.predict(inputs)
        correct = np.sum(predictions == labels)
        accuracy = correct / len(labels)

        return accuracy