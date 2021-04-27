import numpy as np
import em
import common

# X = np.loadtxt("test_incomplete.txt")
# X_gold = np.loadtxt("test_complete.txt")
#
# K = 4
# n, d = X.shape
# seed = 0
#
#
# # TODO: Your code here
# mixture, post = common.init(X, K, seed)
# mixture_new, post_new, log_like = em.run(X, mixture, post)

X = np.loadtxt("netflix_incomplete.txt")
Xgold = np.loadtxt("netflix_complete.txt")
Karray = np.array([12])
seedarray = np.array([1])
logmat = np.zeros((Karray.size,seedarray.size))
for k, value in enumerate(Karray):
    for seed, seedvalue in enumerate(seedarray):
        mixture, post = common.init(X, value, seedvalue)
        mixture_new, post_new, logmat[k, seed] = em.run(X, mixture, post)
        print("Finished processing model for K {} and seed {}".format(value, seedvalue))
print(logmat)

Xpred = em.fill_matrix(X,mixture_new)
mat_pred_error = common.rmse(Xpred,Xgold)
print("RMSE error for predicted matrix is {:.6f}".format(mat_pred_error))

