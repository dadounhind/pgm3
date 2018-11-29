import sklearn
from kmean import KMeans
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class EMGaussian():

    def __init__(self,x, isotropic=False):
        self.x = x
        self.K = 4
        self.d = 2
        self.n = x.shape[0]
        self.isotropic = isotropic
        # tau is a 2D array tau[i,j] = p(Z=j | xi)
        self.tau, self.labels, self.mu = self.init_clusters(x)
        self.update_sigmas()
        self.update_pi()



    def init_clusters(self, x):
        """ Initialse the clusters with a kmeans """
        kmeans = KMeans(self.x, self.K)
        kmeans.fit()
        tau = np.zeros((self.n, self.K))
        labels = kmeans.labels
        for i, label in enumerate(labels):
            tau[i, int(label)] = 1
        return tau, labels, kmeans.centroids

    def update_pi(self):
        """ Update pi , pi(j) = p(Z=j) """
        self.pi = np.sum(self.tau, axis=0)
        self.pi /= self.n

    def update_sigmas(self):
        """ Update sigmas.
        Based on the parameters isotropic compute diagonal matrics (isotropic model)
        or no """

        sigmas = []
        tau = self.tau
        for j in range(self.K):
            mu_j = self.mu[j]
            if not self.isotropic:

                products = [tau[i,j] * np.outer((xi - mu_j), (xi - mu_j))
                            for i,xi in enumerate(self.x)]
                sigmas += [np.sum(products, axis=0)]
                sigmas[j] /= np.sum(tau[:,j])

            else:

                products = [tau[i,j] * np.dot((xi - mu_j), (xi - mu_j))
                            for i,xi in enumerate(self.x)]
                sigma_j = np.sum(products)
                sigma_j /= np.sum(tau[:,j])
                sigma_j /= self.d
                sigmas += [sigma_j * np.eye(self.d, self.d)]


        self.sigmas = sigmas


    def predict(self, xi):
        """ Computes the vector of the probabilities p(Z=j | xi )"""
        probs = np.zeros(self.K)
        mu = self.mu
        pi = self.pi
        sigmas = self.sigmas

        inv_sigmas = [np.linalg.inv(sigmas[j]) for j in range(self.K)]
        det_sigmas = [np.linalg.det(sigmas[j]) for j in range(self.K)]
        for j in range(self.K):

            probs[j] = pi[j] * math.exp(-(xi - mu[j]).T.dot(inv_sigmas[j]).dot(xi - mu[j]) / 2)
            probs[j] /= (2 * math.pi * math.sqrt(det_sigmas[j]))

        probs = probs / np.sum(probs)
        return probs

    def update_mu(self):
        """ Update the centroids """
        mu = []
        for j in range(self.K):
            xi_j = [xi for i,xi in enumerate(self.x) if self.labels[i] == j]
            mu += [np.mean(xi_j)]
        return np.array(mu)

    def fit(self, e=0.005, plot_likelihood=False):
        """ Fit the model, N iterations """
        diff_loss = e + 1
        i = 0
        likeli = [self.compute_neg_likelihood(self.x, self.labels)]

        while np.abs(diff_loss) > e:
            self.tau = np.array([self.predict(xi) for xi in self.x])
            self.labels = np.argmax(self.tau, axis=1)
            self.update_pi()

            for j in range(self.K):

                self.update_mu()
                self.update_sigmas()

            likeli  += [self.compute_neg_likelihood(self.x, self.labels)]
            diff_loss = likeli[i] - likeli[i+1]
            i += 1

        self.labels = np.argmax(self.tau, axis=1)
        print("Fit Gaussian converged after %i iterations" %i)
        if plot_likelihood:
            plt.plot(range(i+1), likeli)
            plt.title("Negative complete likelihood")
            plt.xlabel("Number of iterations")
        return self.labels


    def plot(self):
        """ Plot the points, centroids and covariance matrices """
        fig = plt.figure(0)
        ax = fig.add_subplot(111, aspect='equal')
        plt.scatter(self.x[:,0],self.x[:,1], c=self.labels, s=1)
        plt.scatter(self.mu[:,0], self.mu[:,1], c='black')
        chi_val = 2.146
        for j in range(self.K):
            points = np.array([xi for i,xi in enumerate(self.x)
                                  if self.labels[i] == j])
            cov = self.sigmas[j]
            eig_vals, eig_vec = np.linalg.eig(cov)
            eig_vals = np.sqrt(eig_vals)

            ell = Ellipse(xy=self.mu[j],
                          width=eig_vals[0] * chi_val, height=eig_vals[1] * chi_val,
                          angle=np.rad2deg(np.arccos(eig_vec[0, 0])))

            ell.set_facecolor("none")
            ell.set_edgecolor("blue")
            ax.add_artist(ell)


    def compute_neg_likelihood(self, x, labels):
        """ Compute the - log likelihood / complete likelihood"""
        z_int = np.zeros((self.n, self.K))
        likeli = 0

        for i in range(self.n):
            z_int[i, int(labels[i])] = 1
        log_lik = 0
        sigmas = self.sigmas

        inv_sigmas = [np.linalg.inv(sigmas[j]) for j in range(self.K)]
        det_sigmas = [np.linalg.det(sigmas[j]) for j in range(self.K)]
        for i,xi in enumerate(x):
            j = int(labels[i])

            log_lik -= (xi - self.mu[j]).T.dot(inv_sigmas[j]).dot(xi - self.mu[j]) / 2
            log_lik -= (self.d * math.log(2 * math.pi) + math.log(math.sqrt(det_sigmas[j]))) / 2
            log_lik +=  math.log(self.pi[j])
        return -log_lik
