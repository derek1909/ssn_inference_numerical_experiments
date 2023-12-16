import numpy as np
from parameters import *

def exp_kernel (s2, tau, x1 , x2):
    return s2 * np.exp(-np.sum(np.abs(x1-x2))/tau)


def realistic_kernel (s2, tau, x1 , x2):
    d = np.sum(np.abs(x1-x2))
    return s2 * (1 + d/tau) * np.exp(-d/tau)

def realistic_kernel_w_oscillations (s2, tau, x1, x2, strong, f = 0):
    if strong:
        c1 = 0.3
        c2 = 0.7
    else:
        c1 = 0.8
        c2 = 0.2
    if (f==0):
        return realistic_kernel (s2, tau, x1 , x2)
    else:
        d = np.sum(np.abs(x1-x2))
        return s2 * (c1+c2*np.cos(d*two_pi*f)) * (1 + d/tau) * np.exp(-d/tau)

def compute_cov (s2, tau, s_n2, epsilon, X, strong, f = 0):
    n = len(X[0])
    cov = np.empty([n,n])
    for i in range(n):
        cov[i,i] = s2 + s_n2
        for j in range(i):
            cov[i,j] = realistic_kernel_w_oscillations (s2-epsilon,tau,X[:,i],X[:,j],strong,f)
            cov[j,i] = cov[i,j]

    return cov

def compute_cov_exp (s2, tau, s_n2, epsilon, X):
    n = len(X[0])
    cov = np.empty([n,n])
    for i in range(n):
        cov[i,i] = s2 + s_n2
        for j in range(i):
            cov[i,j] = exp_kernel(s2-epsilon,tau,X[:,i],X[:,j])
            cov[j,i] = cov[i,j]

    return cov
    
def autocorr(x, mean, std):
    x = (x - mean)/std
    result = np.correlate(x, x, "same")/len(x)
    return (result[result.size//2:])
