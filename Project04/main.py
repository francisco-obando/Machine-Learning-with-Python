import numpy as np
import kmeans
import common
import naive_em
import em
from matplotlib import pyplot as plt

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
# #Use common to initialize mixture model for k-means
# Karray = np.arange(1, 4+1)
# seedarray = np.arange(5)
# for K in Karray:
#     for seed in seedarray:
#         mixture, post = common.init(X, K, seed)
#         mixture_new, post_new, cost = kmeans.run(X, mixture,post)
#         common.plot(X, mixture_new, post_new, "K-means with k-{}, seed-{} cost-{:.2f}".format(K, seed, cost))
#

#test naive_em.py implementation
Karray = np.arange(1,4+1)
seedarray = np.arange(5)
logmat=np.zeros((4,5))
for K in Karray:
    for seed in seedarray:
        mixture, post = common.init(X,K,seed)
        mixture_new, post_new, log_like = naive_em.run(X, mixture, post)
        logmat[K-1,seed] = common.bic(X,mixture_new,log_like)
        common.plot(X, mixture_new, post_new,"Naive EM k-{} seed-{} log likelihood = {:.2f}".format(K,seed,log_like))
