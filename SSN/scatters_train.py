# This code computes scatter plots

import numpy as np
import methods as mt
from parameters import *

import datetime, os


def get_correlation(Sigma,std = None):
    if (std == None):
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
    
    in_location = "parameter_files"
    out_location = "results/scatter"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    
    scatter_mu = np.empty([N_pat*N_exc,2])
    scatter_sigma = np.empty([N_pat*N_exc*N_exc,2])

    scatter_var_only = np.empty([N_pat*N_exc,2])
    
    done_mu = 0
    done_std = 0
    done_sigma = 0
    
    print("Start time : ", datetime.datetime.now())
    
    for alpha in range(N_pat):
        mu_gsm = np.loadtxt(in_location+"/mu"+str(alpha))
        mu_ssn = np.loadtxt(in_location+"/mu_evolved_net_"+str(alpha))[:N_exc]
        Sigma_gsm = np.loadtxt(in_location+"/sigma"+str(alpha))
        Sigma_ssn = np.loadtxt(in_location+"/sigma_evolved_net_"+str(alpha))[:N_exc,:N_exc]
         
        scatter_mu[done_mu:done_mu+N_exc,0] = mu_gsm
        scatter_mu[done_mu:done_mu+N_exc,1] = mu_ssn
        scatter_var_only[done_mu:done_mu+N_exc,0] = np.diag(Sigma_gsm)
        scatter_var_only[done_mu:done_mu+N_exc,1] = np.diag(Sigma_ssn)
        done_mu += N_exc
        
        scatter_sigma[done_sigma:done_sigma+N_exc*N_exc,0] = Sigma_gsm.flatten()
        scatter_sigma[done_sigma:done_sigma+N_exc*N_exc,1] = Sigma_ssn.flatten()
        done_sigma += N_exc * N_exc
        
    np.savetxt(out_location+"/scatter_mu_train",scatter_mu)
    np.savetxt(out_location+"/scatter_sigma_train",scatter_sigma)   
    np.savetxt(out_location+"/scatter_var_only_train",scatter_var_only)
       
    print("End time : ", datetime.datetime.now())
   
    
# Call Main
if __name__ == "__main__":
    main()
