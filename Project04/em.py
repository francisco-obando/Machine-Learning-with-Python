"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    #read input data paramters
    n, d= X.shape
    mu = mixture[0]
    var = mixture[1]
    p = mixture[2]
    k = mu.shape[0]
    # eps to prevent underflow of logs
    eps_under = 1e-16

    # Create indicator matrix, from this calculate Cu for each row
    # Create matrix by using 1 whenever entry >0
    delta = X.astype(bool).astype(int)
    #  Find cardinality of each Cu by summing non zero entries of delta matrix
    Cu = np.sum(delta, axis=1)

    # Calculate log posterior probability using auxiliary function f(u,j)
    # l(j|u) = f(u,j) - log(sum_j(f(u,j)))
    # f(u,j) = log(p_j) + log(N(x_u|u_j,sigma_j^2 * I_Cu))
    # add eps to log(p_j) to prevent underflow
    # log(N(x_u|u_j,sigma_j^2) = -(X-mu)^2/(2*sigma^2)-(d/2)*log(2*np.pi*sigma_j^2)
    # select only non zero elements to add to norm by using delta matrix
    lognorm = np.outer(-(Cu/2),np.log(2*np.pi*var)) + (-1/2)*(np.linalg.norm((X[:, np.newaxis]-mu)*delta[:, np.newaxis], 2, axis=2)**2)/var
    f = np.log(p+eps_under) + lognorm
    log_post = f - logsumexp(f,axis=1).reshape((n, 1))

    # calculate posterior probabilities
    # p(j|u) = p_j * N(x_Cu|mu_j,sigma_j^2)/sum_t(N(x_Cu|mu_t,sigma_t^2))
    # place holder for soft count NxK matrix
    post = np.exp(log_post)
    # log likelihood is sum of logs of probabilities, which are the exponentials of the log_post variable
    # to calculate probabilities use the logsumexp function to sum for all K categories and add for all observations
    log_like = np.sum(logsumexp(f,axis=1).reshape((n,1)),axis=0)

    return post, float(log_like)



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    #read parameters from data and model
    n, d = X.shape
    mu = mixture[0]
    var = mixture[1]
    p = mixture[2]
    K = mu.shape[0]

    # create indicator matrix
    # count non zero entries for each observation
    delta = X.astype(bool).astype(int)
    Cu = np.sum(delta,axis=1)

    # Calculate updated parameters
    # Calculate updated component probabilities in log space
    log_p = np.log(np.sum(post, axis =0))-np.log(n)
    p_hat = np.exp(log_p)

    # Calculate updated means
    # Implement check sum_t(p(t|u)delta(l|Cu))>1
    # mu_denominator, mu_update has size K x d,
    mu_numerator = post.T @ X
    mu_denominator = post.T @ delta
    mu_update = np.where(mu_denominator>=1, 1, 0)
    # Use mu_update to mask mean update
    # update only if denominator mask is greater than 1, if not keep original mu,
    # np.where((mu_numerator/mu_denominator)*mu_update==0,mu,mu_numerator/mu_denominator)
    mu_hat = np.where(np.divide(mu_numerator,mu_denominator,out=np.zeros_like(mu_numerator),where=mu_denominator!=0)*mu_update == 0,mu,
                      np.divide(mu_numerator,mu_denominator,out=np.zeros_like(mu_numerator),where=mu_denominator!=0))

    # Calculate updated variances
    # Calculate deviation from K means masking with observed data
    var_numerator = np.sum(post * np.linalg.norm((X[:, np.newaxis]-mu_hat)*delta[:, np.newaxis], 2, axis=2)**2, axis=0)
    var_denominator = np.sum(post * Cu.reshape((n,1)),axis=0)
    var_hat = np.divide(var_numerator, var_denominator, out=np.zeros_like(var_numerator),where=var_denominator != 0)
    # Ensure the minimum variance for each component
    var_hat = np.where(var_hat>min_variance, var_hat, min_variance)

    return GaussianMixture(mu_hat, var_hat, p_hat)


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
    prev_loglike = None
    loglike = None
    eps = 1e-6

    while (prev_loglike is None or np.abs((loglike-prev_loglike)/loglike)>eps):
        prev_loglike = loglike
        post, loglike = estep(X,mixture)
        mixture = mstep(X,post,mixture)
    return mixture, post, loglike


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    # Perform expectation step to obtain posterior from mixture model
    #read input data paramters
    n, d= X.shape
    mu = mixture[0]
    var = mixture[1]
    p = mixture[2]
    k = mu.shape[0]
    # eps to prevent underflow of logs
    eps_under = 1e-16

    # Create indicator matrix, from this calculate Cu for each row
    # Create matrix by using 1 whenever entry >0
    delta = X.astype(bool).astype(int)
    #  Find cardinality of each Cu by summing non zero entries of delta matrix
    Cu = np.sum(delta, axis=1)

    # Calculate log posterior probability using auxiliary function f(u,j)
    # l(j|u) = f(u,j) - log(sum_j(f(u,j)))
    # f(u,j) = log(p_j) + log(N(x_u|u_j,sigma_j^2 * I_Cu))
    # add eps to log(p_j) to prevent underflow
    # log(N(x_u|u_j,sigma_j^2) = -(X-mu)^2/(2*sigma^2)-(d/2)*log(2*np.pi*sigma_j^2)
    # select only non zero elements to add to norm by using delta matrix
    lognorm = np.outer(-(Cu/2),np.log(2*np.pi*var)) + (-1/2)*(np.linalg.norm((X[:, np.newaxis]-mu)*delta[:, np.newaxis], 2, axis=2)**2)/var
    f = np.log(p+eps_under) + lognorm
    log_post = f - logsumexp(f,axis=1).reshape((n, 1))

    # calculate posterior probabilities
    # p(j|u) = p_j * N(x_Cu|mu_j,sigma_j^2)/sum_t(N(x_Cu|mu_t,sigma_t^2))
    # place holder for soft count NxK matrix
    post = np.exp(log_post)


    # indicate index where no information is available
    update_indicator = np.where(X != 0,0,1)
    # Calculate predicted values based on model means N x d matrix
    predicted = post @ mu
    # Keep predicted values only when the original data does not contain it
    X_pred = np.where(update_indicator == 1, predicted, X)
    return X_pred
