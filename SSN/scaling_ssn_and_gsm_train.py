# Compute scaling of the gsm and ssn moments for the training set

import numpy as np
import scipy as sp
from parameters import *
import sys, os

############
#   Main   #
############

def main():
    
    use_full_net = True  # If false, compute scaling using ADF instead
    in_location = "parameter_files"
    out_location = "results/scaling"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
        
    z_array = np.array([0.0,0.125,0.25,0.5,1.0])
           
    results_ssn = np.empty([N_pat,5])
    results_gsm = np.empty([N_pat,5])
    
    for alpha in range(N_pat):
        
        h = np.loadtxt(in_location+"/h_true_"+str(alpha)+"_learn")
        
        mu_gsm = np.loadtxt(in_location +"/mu"+str(alpha))
        std_gsm = np.loadtxt(in_location +"/std"+str(alpha))
        
        if (use_full_net):
            mu_ssn = np.loadtxt(in_location +"/mu_evolved_net_"+str(alpha))
            std_ssn = np.loadtxt(in_location +"/std_evolved_net_"+str(alpha))
        else:
            mu_ssn = np.loadtxt(in_location +"/mu_evolved_"+str(alpha))
            std_ssn = np.loadtxt(in_location +"/std_evolved_"+str(alpha))
        
        results_ssn[alpha][0] = z_array[alpha]
        results_ssn[alpha][1] = np.average(h[:N_exc])
        results_ssn[alpha][2] = np.average(mu_ssn[:N_exc])
        results_ssn[alpha][3] = np.amax(mu_ssn[:N_exc])
        results_ssn[alpha][4] = np.average(std_ssn[:N_exc])
        
        results_gsm[alpha][0] = z_array[alpha]
        results_gsm[alpha][1] = np.average(h[:N_exc])
        results_gsm[alpha][2] = np.average(mu_gsm[:N_exc])
        results_gsm[alpha][3] = np.amax(mu_gsm[:N_exc])
        results_gsm[alpha][4] = np.average(std_gsm[:N_exc])
        
    if (use_full_net):
        np.savetxt(out_location+"/mu_and_std_vs_contrast_ssn_full_net_train",results_ssn)
    else:
        np.savetxt(out_location+"/mu_and_std_vs_contrast_ssn_train",results_ssn)
    
    np.savetxt(out_location+"/mu_and_std_vs_contrast_nl_transformed_gsm_train",results_gsm)
            

# Call Main
if __name__ == "__main__":
    main()
