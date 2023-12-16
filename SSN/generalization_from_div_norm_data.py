# This code takes the data used for div norm training and reuses it to check 
# generalization from GSM to SSN

import numpy as np
import methods as mt
from parameters import *

import datetime, os

def get_correlation(Sigma):
    std = np.sqrt(np.diag(Sigma))
    Corr = np.empty([N_exc,N_exc])    
    for i in range(N_exc):
        for j in range(N_exc):
            Corr[i,j] = Sigma[i,j]/(std[i]*std[j])
    return Corr

############
#   Main   #
############

def main():
    
    in_location = "results/div_norm_data/z_from_gamma/test"
    
    out_location = in_location + "/../../../generalization/z_from_gamma/"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    
    N_contrasts = 5
    N_points_per_contrast = 20
    
    max_points_scatter = 2 * 50 * 50
    
    N_points = N_points_per_contrast * N_contrasts
    
    scatter_mu = np.empty([N_points*N_exc,3])
    scatter_std = np.empty([N_points*N_exc,3])
    scatter_sigma = np.empty([N_points*N_exc*N_exc,3])
    
    done_mu = 0
    done_std = 0
    done_sigma = 0
    
    print("Start time : ", datetime.datetime.now())
    
    for s in range (N_points_per_contrast): 
        for alpha in range(N_contrasts):
            index = s * N_contrasts + alpha
            z = np.loadtxt(in_location+"/z_"+str(index))
            mu_gsm = np.loadtxt(in_location+"/mu_GSM_"+str(index))
            mu_ssn = np.loadtxt(in_location+"/mu_SSN_"+str(index))[:N_exc]
            std_gsm = np.loadtxt(in_location+"/std_GSM_"+str(index))
            std_ssn = np.loadtxt(in_location+"/std_SSN_"+str(index))[:N_exc]
            Sigma_gsm = np.loadtxt(in_location+"/Sigma_GSM_"+str(index))
            Sigma_ssn = np.loadtxt(in_location+"/Sigma_SSN_"+str(index))[:N_exc,:N_exc]
             
            Correlation_gsm = get_correlation(Sigma_gsm)
            Correlation_ssn = get_correlation(Sigma_ssn)
            np.savetxt(in_location+"/Corr_GSM_"+str(index),Correlation_gsm)
            np.savetxt(in_location+"/Corr_SSN_"+str(index),Correlation_ssn)
            
            scatter_mu[done_mu:done_mu+N_exc,0] = mu_gsm
            scatter_mu[done_mu:done_mu+N_exc,1] = mu_ssn
            scatter_mu[done_mu:done_mu+N_exc,2] = z
            done_mu += N_exc
            
            scatter_std[done_std:done_std+N_exc,0] = std_gsm
            scatter_std[done_std:done_std+N_exc,1] = std_ssn
            scatter_std[done_std:done_std+N_exc,2] = z
            done_std += N_exc
            
            scatter_sigma[done_sigma:done_sigma+N_exc*N_exc,0] = Sigma_gsm.flatten()
            scatter_sigma[done_sigma:done_sigma+N_exc*N_exc,1] = Sigma_ssn.flatten()
            scatter_sigma[done_sigma:done_sigma+N_exc*N_exc,2] = z
            done_sigma += N_exc * N_exc
            
    if (done_mu > max_points_scatter):
        print("subsampling all scatters")
        rand_mu_std = np.random.randint(0, high=done_mu,size=max_points_scatter)
        rand_sigma =  np.random.randint(0, high=done_sigma,size=max_points_scatter)
        scatter_mu = scatter_mu[rand_mu_std]
        scatter_std = scatter_std[rand_mu_std]
        scatter_sigma = scatter_sigma[rand_sigma]
            
    elif (done_sigma > max_points_scatter):   
        print("subsampling only sigma scatter")
        rand_sigma =  np.random.randint(0, high=done_sigma, size=max_points_scatter)
        scatter_sigma = scatter_sigma[rand_sigma]
    
    np.savetxt(out_location+"/scatter_mu",scatter_mu)
    np.savetxt(out_location+"/scatter_std",scatter_std)
    np.savetxt(out_location+"/scatter_sigma",scatter_sigma)   
    
       
    print("End time : ", datetime.datetime.now())
   
    
# Call Main
if __name__ == "__main__":
    main()
