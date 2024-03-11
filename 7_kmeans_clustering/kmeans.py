import numpy as np
import random

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    Hint: use random.sample (it is already imported for you!)
    :param k: number of cluster centroids
    :param inputs: a 2D Numpy array, each row of which is one input
    :rand: random seed to be used when sampling from inputs
    :return: a Numpy array of k cluster centroids, one per row
    """
    # Hint: Use random.sample
    # TODO
    centroids = np.array(random.sample(inputs.tolist(), k))
    return centroids


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    :param inputs: inputs of data, a 2D Numpy array
    :param centroids: a Numpy array of k current centroids
    :return: a Numpy array of centroid indices, one for each row of the inputs
    """
    # TODO
    indices = np.zeros(inputs.shape[0])
    for i in range(len(inputs)):
        dist = np.linalg.norm(inputs[i] - centroids, axis=1)
        indices[i] = np.argmin(dist)
    
    return indices
        


def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    :param inputs: inputs of data, a 2D Numpy array
    :param indices: a Numpy array of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    centroids = np.zeros((k, inputs.shape[1]))
   
    for i in range(k):
        centroids[i] = np.mean(inputs[indices == i], axis=0)
    return centroids
        


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    :param inputs: inputs of data, a 2D Numpy array
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: the tolerance we determine convergence with when compared to the ratio as stated on handout
    :param rand: a given random seed to be used within init_centroids
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    

    centroids = init_centroids(k, inputs)
    

    for i in range(max_iter):
        indices = assign_step(inputs, centroids)
        new_centroids = update_step(inputs, indices, k)
        
        # Check 
        diff = np.linalg.norm(new_centroids - centroids)
        ratio = diff / np.linalg.norm(centroids)
        if ratio < tol:
            break
            
        centroids = new_centroids
        
    return centroids


class KmeansClassifier(object):
    """
    K-Means Classifier via Iterative Improvement
    @attrs:
        k: The number of clusters to form as well as the number of centroids to
           generate (default = 10), an int
        tol: Value specifying our convergence criterion. If the ratio of the
             distance each centroid moves to the previous position of the centroid
             is less than this value, then we declare convergence.
        max_iter: the maximum number of times the algorithm can iterate trying
                  to optimize the centroid values, an int,
                  the default value is set to 500 iterations
        cluster_centers_: a Numpy array where each element is one of the k cluster centers
    """

    def __init__(self, n_clusters = 10, max_iter = 500, threshold = 1e-6):
        """
        Initiate K-Means with some parameters
        """
        self.k = n_clusters
        self.tol = threshold
        self.max_iter = max_iter
        self.cluster_centers_ = np.array([])

    def train(self, X):
        """
        Compute K-Means clustering on each class label and store your result in self.cluster_centers_
        :param X: inputs of training data, a 2D Numpy array
        :param rand: random seed to be used during training
        :return: None
        """
        self.cluster_centers_ = kmeans(X, self.k, self.max_iter, self.tol)

        

    def predict(self, X, centroid_assignments):
        """
        Predicts the label of each sample in X based on the assigned centroid_assignments.
        :param X: A dataset as a 2D Numpy array
        :param centroid_assignments: a Numpy array of 10 digits (0-9) representing the interpretations of the digits of the plotted centroids
        :return: A Numpy array of predicted labels
        """

        # TODO: complete this step only after having plotted the centroids!
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            centroids_i = self.cluster_centers_[i]
            distances[:, i] = np.linalg.norm(X - centroids_i, axis=1)
        
        pred = np.argmin(distances, axis=1)
        for i in range(len(pred)):
            pred[i] = centroid_assignments[pred[i]]

        return pred

    def accuracy(self, data, centroid_assignments):
        """
        Compute accuracy of the model when applied to data
        :param data: a namedtuple including inputs and labels
        :return: a float number indicating accuracy
        """
        pred = self.predict(data.inputs, centroid_assignments)
        return np.mean(pred == data.labels)


