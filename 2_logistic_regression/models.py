import random
import numpy as np

def softmax(x):
    '''
    Apply softmax to an array
    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)

class LogisticRegression:
    '''
    Multiclass Logistic Regression that learns weights using 
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.
        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_classes, n_features + 1))  # An extra row added for the bias
        self.alpha = 0.03  # DO NOT TUNE THIS PARAMETER
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        
        converge = False
        epoch = 0
         
    
        while not converge:
            epoch += 1
            data = list(zip(X,Y))
            np.random.shuffle(data)
            X,Y = zip(*data)
     
    
            last_epoch = self.loss(X,Y)
            
            for i in range(len(X)// self.batch_size):
                X_b = X[i*self.batch_size:(i+1)*self.batch_size]
                Y_b = Y[i*self.batch_size:(i+1)*self.batch_size]
                
                grad_w = np.zeros(self.weights.shape) 
                for (x, y) in zip(X_b, Y_b):
                    for j in range(self.n_classes):
                        if y == j:
                            grad_w[j] += (softmax(np.matmul(self.weights, x))[j]- 1) * x  
                        else:
                            grad_w[j] += softmax(np.matmul(self.weights, x))[j] * x
                self.weights = self.weights - (self.alpha*grad_w)/self.batch_size
            this_epoch_loss = self.loss(X, Y)
            abs_mean = this_epoch_loss - last_epoch
            if abs_mean < self.conv_threshold:
                converge = True
    
        return epoch
            


    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        logloss = 0
        for i in range(len(Y)):
            predict = softmax(np.matmul(self.weights, X[i]))
            for j in range(self.n_classes):
                if Y[i] == j:
                    logloss -= np.log(predict[j])
        return logloss/len(Y)
                    



    def predict(self, X):
        '''
        Compute predictions based on the learned weigths and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        prediction = np.zeros(len(X))
        for i in range(len(X)): 
            prediction[i] = np.argmax(softmax(np.matmul(self.weights, X[i])))
            print(prediction[i],softmax(np.matmul(self.weights, X[i])))
        return prediction



    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        Y_pred = self.predict(X)
        accuracy = sum(Y_pred == Y)/len(Y)
    
        return accuracy