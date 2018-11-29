import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import random

class KMeans():

    def __init__(self, x, K):

        self.x = x
        self.K = K
        self.n = x.shape[0]
        self.centroids = self.x[random.sample(range(self.n),K)]
        self.labels = self.predict(x)
        self.loss = self.compute_loss()


    def predict(self, x):
        """ Find closer centroids and ssign a label to each xi in x"""
        centroids = self.centroids
        labels = np.zeros(len(x))
        for i,xi in enumerate(x):
            distances = [np.linalg.norm(xi - c) for c in centroids]
            labels[i] = int(np.argmin(distances))

        return labels

    def update_centroids(self):
        """ Update the centroids based on the assignements """
        for k in range(self.K):
            points = np.array([xi for i,xi in enumerate(self.x)
                                  if self.labels[i] == k])

            self.centroids[k] = np.mean(points, axis=0)

    def compute_loss(self):
        """ Compute the loss : sum of squares distances to centroids """
        centroids = self.centroids
        labels = self.labels
        dist = [np.linalg.norm(xi - centroids[int(labels[i])]) ** 2
                for i,xi in enumerate(self.x)]
        return np.sum(dist)

    def fit(self, e=0.01):
        """ Fit the model. Iterates until the difference in the loss is
        lower than e """
        diff_loss = e + 1
        i = 0
        errors = []
        loss = self.loss
        while diff_loss > e:

            self.update_centroids()
            self.labels = self.predict(self.x)

            new_loss = self.compute_loss()
            diff_loss, loss = abs(loss - new_loss), new_loss
            i += 1

        print("Kmeans converged after %i iterations, loss: %f" %(i, loss))
        self.loss = loss

    def plot(self):
        """ Plot the points and centroids """
        plt.scatter(self.x[:,0],self.x[:,1], c=self.labels, s=1)
        plt.scatter(self.centroids[:,0], self.centroids[:,1], c='black')
        plt.title("Kmean Algorithm - number of cluster= %i" %self.K)
