# This code evolves the network forward in time, collecting samples, 
# to then compute autocorrelation and power spectrum

import numpy as np
import scipy as sp
import methods as mt
from parameters import *

import datetime, os


############
#   Main   #
############

def main():
    in_location = in_location = "parameter_files"
    
    custom_contrasts = False
    
    if (custom_contrasts == True):
        out_location = "results/correlations_and_power_spectrum_custom_contrasts"
    else:
        out_location = "results/correlations_and_power_spectrum"
        
    if not os.path.exists(out_location):
        os.makedirs(out_location)  
        
    epsilon = 1.0E-12
    
    #----------------------------------------------#
    #   We import W, h, and the noise covariance   #
    #----------------------------------------------#

    Sigma_eta = np.loadtxt(in_location+"/sigma_eta_learn")
        
    if (custom_contrasts):
        input_scaling_learn = np.loadtxt(in_location+"/input_scaling_learn")
        input_baseline_learn = np.loadtxt(in_location+"/input_baseline_learn")
        input_nl_pow_learn = np.loadtxt(in_location+"/input_nl_pow_learn")
        h_full_contrast = np.loadtxt(in_location+"/h4")
        
        N_pat = 8
        z = np.logspace(-4, 1, num=N_pat, endpoint=True, base=2.0)
        h = np.empty([N_pat,N])
        for alpha in range(N_pat):
            h_lin= z[alpha] * h_full_contrast
            h[alpha] = input_scaling_learn * np.power((h_lin + input_baseline_learn),input_nl_pow_learn)
    else:
        N_pat = 5
        h = np.empty([N_pat,N])
        for alpha in range(N_pat):
            h[alpha] = np.loadtxt(in_location+"/h_true_"+str(alpha)+"_learn")
        
    W = np.loadtxt(in_location+"/w_learn")
       
    #------------------------------#
    #   Evolution of the network   #
    #------------------------------#
           
    # We take simple initial conditions for mu and Sigma
    mu_0 = np.empty([N_pat,N])
    Sigma_0 = np.empty([N_pat,N,N])
           
    for alpha in range(N_pat):
        if (custom_contrasts):
            mu_0[alpha]  = np.zeros(N)
            Sigma_0[alpha] = 4.0*np.identity(N) 
        
        else:
            mu_0[alpha] = np.loadtxt(in_location+"/mu_evolved_net_"+str(alpha))
            Sigma_0[alpha] = np.loadtxt(in_location+"/sigma_evolved_net_"+str(alpha))
            
        
    mu_final = np.empty([N_pat,N])
    Sigma_final = np.empty([N_pat,N,N])
    
    total_time = 50000.0 * tau_e # 1000s
    t_bet_samp = 2 * dt
    steps_bet_samp = int(t_bet_samp/dt)  
    #total_time = 10000.0 * tau_e # 400s
    #t_bet_samp = dt
    #steps_bet_samp = 1  
    sample_size = int(total_time/t_bet_samp)
    
    print("Start time : ", datetime.datetime.now())
    
    r_samples = np.empty([sample_size,N])
                 
    for alpha in range(N_pat):
        u0 = np.random.multivariate_normal(mean = mu_0[alpha], cov = Sigma_0[alpha])
        eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
        
        print("evolving pattern" + str(alpha))
        u, eta = mt.network_evolution(W,h[alpha],u0,Sigma_eta, eta = eta0)
        print("sampling from the network")
        (u_samples,eta_samples,_,_) = mt.network_sample(W,h[alpha],u,eta,sample_size,
                                                steps_bet_samp,Sigma_eta)
        
        r_samples = mt.get_r(u_samples)
            
        np.save(out_location+"/u_samples_"+str(alpha),u_samples)
        np.save(out_location+"/r_samples_"+str(alpha),r_samples)
        np.save(out_location+"/eta_samples_"+str(alpha), eta_samples)
                                                                      
    print("End time : ", datetime.datetime.now())
        

# Call Main
if __name__ == "__main__":
    main()
