"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # extract components of mixture model
    # (mu, var, p)
    # n number of data points
    # d number of features
    # K number of clusters

    # np array with K means for d features K x d
    mu = mixture[0]
    # np array with K variances
    var = mixture[1]
    # np array with K probabilities, one for each component
    p = mixture[2]

    n, d = X.shape
    K, _ = mu.shape

    #calculate posterior probabilities / soft count
    # p(j|i) = probability (x_i came from component j| x_i)
    # p(j|i) = p_j * N(x_i; mu_j, var_j)/sum_t(p_t * N(x_i; mu_t, var_t))
    # p(j|i) is a n x K matrix
    # implement the general case with non diagonal covariance matrix
    """
    Calculate multivariate normal probability density function N(X|mu,covariance)
    :param X: point to calculate pdf - d dimensional vector
    :param mu: mean data point for distribution - d dimensional vector
    :param covariance: covariance matrix - dxd dimensional vector
    :return: pdf evaluated at point X
    """
    # Create place holder for posterior probabilities
    post = np.zeros((n,K))
    # Cast K means for observed data (N x K x d)
    Xmeans = X[:,np.newaxis] - mu
    # Cast K covariance matrices (K x d x d)
    if var.ndim == 1:
        covariance = var[:, np.newaxis, np.newaxis] * np.identity(d)
    else:
        covariance = var

    #calculate determinant of each covariance matrix
    #calculate matrix inverse for each covariance matrix
    determinant = np.zeros(K)
    covariance_inv = np.zeros((K,d,d))
    for i in range(K):
        determinant[i] = np.linalg.det(covariance[i, :, :])
        covariance_inv[i, :, :] = np.linalg.pinv(covariance[i, :, :])

    #use determinant of covariance matrices to find normalizing constants
    determinant = (determinant **(-1/2)) * (2*np.pi) **(-d/2)
    #einstein sum over tensors to reduce to N x K matrix (x-u)T simga_inv (x-u)
    post = np.einsum('k...i,...ji,k...i->k...', Xmeans, covariance_inv, Xmeans)
    post = np.exp(-post/2)
    # use mixture probabilities for each component
    for i in range(K):
        post[:,i] = p[i] * determinant [i] * post[:,i]
    #post = p * (determinant * post)
    # sum to calculate log likelihood and normalize sum(p_j N(xi,mu_j,sigma_j^2) for fixed i
    total = np.sum(post, axis = 1)
    post = np.divide(post,total[:,np.newaxis])

    loglike = np.sum(np.log(total),axis = 0)
    return post, float(loglike)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # n number of data points
    # d number of features
    # K number of clusters
    n, d = X.shape
    _, K = post.shape

    #update model parameters with calculated posterior probabilities
    # find the updated component probabilities (K vector)
    p = (np.sum(post, axis=0)/ np.sum(post))
    # find the updated K means of the model (K x d) vector
    mu = (1/np.sum(post, axis=0).reshape(K,1))*(post.T @ X)
    Xnew = X[:, np.newaxis] - mu
    # find the updated covariance matrices (k x d x d)
    #var = np.einsum('k...,k...i,k...j->...ij',post,Xnew,Xnew)
    norm = np.power(np.linalg.norm(Xnew,2,axis=2),2)
    suma = np.sum(post * norm, axis=0)
    var = (1/(np.sum(post, axis=0)*d))*suma
    #norm = 1 / (np.sum(post, axis=0))
    #var = var * norm[:, np.newaxis, np.newaxis]
    # if implementing only iid diagonal GMM K vector
    #var = var[:, 0, 0]

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    # previous log likelihood
    # first step log likelihood
    # relative change threshold
    prev_loglike = None
    loglike = None
    eps = 1e-6
    while (prev_loglike is None or np.abs((loglike-prev_loglike)/loglike) > eps):
        prev_loglike = loglike
        post, loglike = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, loglike
