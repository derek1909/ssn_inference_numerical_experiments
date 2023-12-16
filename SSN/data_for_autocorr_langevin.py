# This code evolves the network forward in time, collecting samples, 
# to then compute autocorrelation and power spectrum for the Langevin case

import numpy as np
import scipy as sp
import methods as mt
from parameters import *

import datetime, os


############
#   Main   #
############

def main():
    
    in_location_net = "parameter_files"
    in_location_langevin = "results/langevin"
    out_location = in_location_langevin + "/autocorr"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
        
    epsilon = 1.0E-12
    
    #--------------------------#
    #   We import parameters   #
    #--------------------------#

    mu_net = np.empty([N_pat,N])
    Sigma_net = np.empty([N_pat,N,N])
    A = np.empty([N_pat,N,N])
    
    B = np.loadtxt(in_location_langevin+"/B")
    Sigma_eta = tau_n_inv * B
    
    for alpha in range(N_pat):
        mu_net[alpha] = np.loadtxt(in_location_net+"/mu_evolved_net_"+str(alpha))
        Sigma_net[alpha] = np.loadtxt(in_location_net+"/sigma_evolved_net_"+str(alpha))
        A[alpha] = np.loadtxt(in_location_langevin+"/A_"+str(alpha))
    
       
    #------------------------------#
    #   Evolution of the network   #
    #------------------------------#
           
    # We take simple initial conditions for mu and Sigma
    
    total_time = 50000.0 * tau_e # 1000s
    t_bet_samp = 2*dt
    steps_bet_samp = int(t_bet_samp/dt)    
    sample_size = int(total_time/t_bet_samp)
    
    print("Start time : ", datetime.datetime.now())
    r_samples = np.empty([sample_size,N])
    
            
    for alpha in range(N_pat):
        print("sampling pattern " + str(alpha))
        mu_ext = np.concatenate((mu_net[alpha],np.zeros(N)))
        Sigma_solve = np.loadtxt(in_location_langevin+"/Sigma_solve_"+str(alpha))
        init = np.random.multivariate_normal(mean = mu_ext, cov = Sigma_solve)
        u0 = init[:N]
        eta0 = init[N:]
        (u_samples,_,_,_) = mt.Langevin_sample(A[alpha],B,mu_net[alpha],
                                           sample_size,steps_bet_samp,
                                           u0,eta0)
        
        for i in range(sample_size):
            r_samples[i] = mt.get_r(u_samples[i])
            
               
        np.save(out_location+"/u_samples_"+str(alpha),u_samples)
        np.save(out_location+"/r_samples_"+str(alpha),r_samples)
                                                           
    
    print("End time : ", datetime.datetime.now())
        

# Call Main
if __name__ == "__main__":
    main()
