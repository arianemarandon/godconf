import numpy as np 


def sample_mixture(n_samples, nullprop, P0, P1):
    """
    Sample from a two-component mixture model:      nullprop * P0 + (1-nullprop) * P1
    
    P0 / P1 must have a .rvs() method as in scipy.stats classes 
    e.g. P0 = scipy.stats.multivariate_normal(mean=0, cov=1) is N(0,1)
    """

    n_null = np.random.binomial(n_samples, nullprop)

    xnull = P0.rvs(size = n_null)
    xnonull = P1.rvs(size = n_samples-n_null)
    
    y = np.concatenate([np.zeros(n_null), np.ones(n_samples-n_null)])
    x = np.concatenate([xnull, xnonull])
    
    #shuffle 
    ind=np.random.permutation(n_samples)     
    return x[ind], y[ind]


def sample_det_mixture(n_samples, nullprop, P0, P1):
    """
    Sample from P0 and P1 in fixed sizes 
    """

    n_null = int(n_samples * nullprop)

    xnull = P0.rvs(size = n_null)
    xnonull = P1.rvs(size = n_samples-n_null)
    
    y = np.concatenate([np.zeros(n_null), np.ones(n_samples-n_null)])
    x = np.concatenate([xnull, xnonull])
    
    #shuffle 
    ind=np.random.permutation(n_samples)     
    return x[ind], y[ind]


def get_fdp(ytrue, rejection_set):
    """
    ytrue: vector of size m indicating for each test point whether it is a null (0) or a non-null (1) 
    rejection_set: a list of the indexes (corresponding to rejections of some procedure) 

    Return: the FDP and the TDP for the rejection set <rejection_set>
    """

    if rejection_set.size:
        fdp = np.sum(ytrue[rejection_set] == 0) / len(rejection_set)
        tdp = np.sum(ytrue[rejection_set] == 1) / np.sum(ytrue==1)
    else: 
        fdp=0
        tdp=0
    return fdp, tdp
