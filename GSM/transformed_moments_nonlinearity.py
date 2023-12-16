'''
This code takes the GSM moments and finds the corresponding ones for the non-linear model

Created on Dec 17, 2017

@author: Rodrigo

'''

import math
import time
from time import localtime, strftime
import numpy as np
import scipy as sp
from scipy.stats import norm, multivariate_normal

import os, sys

#np.random.seed(12345)
# General Methods

N_exc = 50

def np_th(u):
    return np.maximum(u,0)
  
def nl_fun(u,nl_scale,nl_baseline,nl_power):
    return nl_scale*(np_th(u+nl_baseline)**nl_power)

def error(x,y):
    return np.square(x-y).mean()
    
def get_mean_var(points_1d,mu,std,nl_scale,nl_baseline,nl_power):
    new_mean = np.empty([N_exc])
    new_var = np.empty([N_exc])
    for i in range(N_exc):
        mu_i = mu[i]
        std_i = std[i]    
        points_resc = std_i * points_1d + mu_i
        values_dist = norm.pdf(points_resc, loc = mu_i, scale = std_i)
        norm_f = np.sum(values_dist)
        values_fun = nl_fun(points_resc, nl_scale,nl_baseline,nl_power)
        new_mean[i] = np.sum(values_dist*values_fun)/norm_f
        new_var[i] = np.sum(values_dist*(values_fun-new_mean[i])**2.0)/norm_f    
    return (new_mean, new_var)

def get_cov(points_1d,mu,std,Cov,new_mean,new_var,nl_scale,nl_baseline,nl_power):
    new_Cov = np.empty([N_exc,N_exc])
    n_points = len(points_1d)
    for i in range(N_exc):
        new_Cov[i,i] = new_var[i]
        for j in range(i+1,N_exc):
            mu_i = mu[i]
            std_i = std[i]
            mu_j = mu[j]
            std_j = std[j]
            mean_red = np.array([mu_i,mu_j])    
            Cov_red = np.array([[Cov[i,i],Cov[i,j]],[Cov[j,i],Cov[j,j]]])
            points_resc_i = std_i * points_1d + mu_i
            points_resc_j = std_j * points_1d + mu_j
            X,Y = np.meshgrid(points_resc_i, points_resc_j)
            points_2d = np.vstack((X.flatten(), Y.flatten())).T
            values_dist = multivariate_normal.pdf(points_2d, 
                                    mean=mean_red,cov=Cov_red).reshape(n_points,n_points)
            
            norm_f = np.sum(values_dist)
            values_fun_2 = np.outer(nl_fun(points_resc_i, nl_scale,nl_baseline,nl_power)-new_mean[i],
                                  nl_fun(points_resc_j, nl_scale,nl_baseline,nl_power)-new_mean[j])
            new_Cov[i,j] = np.sum(values_dist*values_fun_2)/norm_f
    
    for i in range(N_exc):
        for j in range(0,i):
            new_Cov[i,j] = new_Cov[j,i]
                    
    return new_Cov


       
############
#   Main   #
############

def main():
      
    # File locations    
    in_location = "generalization_data/no_noise/results_GSM"
    out_location = "generalization_data/no_noise/results_GSM_nl"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
            
    epsilon = 1.0E-12
            
    nl_scale = 2.4
    nl_power = 0.6
    nl_baseline = (3.5/nl_scale)**(1.0/nl_power)
            
    
    points_1d = np.linspace(-4.0,4.0, num = 201)
    
    N_pat = 5
    
    n_points = 200 * N_pat
       
    for alpha in range(n_points):
        print(alpha)
        # Loading moments
        z = np.loadtxt(in_location+"/z_"+str(alpha))
        x = np.loadtxt(in_location+"/x_"+str(alpha))
        y = np.loadtxt(in_location+"/y_"+str(alpha))
        h_true = np.loadtxt(in_location+"/h_true_"+str(alpha))
        mu_GSM = np.loadtxt(in_location+"/mu_true_z_"+str(alpha))[:N_exc]
        Sigma_GSM = np.loadtxt(in_location+"/Sigma_true_z_"+str(alpha))[:N_exc,:N_exc]
        var_GSM = np.diag(Sigma_GSM)
        std_GSM = np.sqrt(var_GSM)
                    
        
        (mu_new, var_new) = get_mean_var(points_1d,mu_GSM,std_GSM,nl_scale,nl_baseline,nl_power)
        Sigma_new = get_cov(points_1d,mu_GSM,std_GSM,Sigma_GSM,
                            mu_new,var_new,nl_scale,nl_baseline,nl_power)
        
        np.savetxt(out_location+"/x_"+str(alpha),x)
        np.savetxt(out_location+"/y_"+str(alpha),y)
        np.savetxt(out_location+"/z_"+str(alpha),np.expand_dims(z, axis = 0))
        np.savetxt(out_location+"/h_true_"+str(alpha),h_true)        
        np.savetxt(out_location+"/mu_true_z_"+str(alpha),mu_new)
        np.savetxt(out_location+"/std_true_z_"+str(alpha), np.sqrt(var_new))
        np.savetxt(out_location+"/Sigma_true_z_"+str(alpha), Sigma_new)
        
            
    
# Call Main
if __name__ == "__main__":
    main()  
