# This code takes a weight matrix and a set of inputs and computes the stationary
# solution for the output moments using the full network simulation for the Langevin case

import numpy as np
import scipy as sp
import methods as mt
from parameters import *

import datetime, os

# This method makes sure that a cov matrix is pos def
def regularize_Sigma(Sigma,location,name,epsilon):
    # We check that Sigma is pos def
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    ei_min = np.amin(eigenvalues.real)
    
    print("ei_min of "+name+": ", ei_min)
    if (ei_min <= 0.0):
        print("Regularizing "+name+"...")
        Sigma += (epsilon - ei_min) * np.identity(N)
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        ei_min = np.amin(eigenvalues.real)
        print("New ei_min of "+name+": ", ei_min)
        if ei_min > 0.0:
            print("reg successful")
            np.savetxt(location+"/"+name+"_reg",Sigma)
        else:
            print("reg failed")
    else:
        print(name+" was already pos def")
        
    return Sigma

############
#   Main   #
############

def main():
    in_location_net = "results"
    in_location_langevin = "results/langevin"
    out_location = in_location_langevin + "/moments"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    
    epsilon = 1.0E-12
    
    sample_size = 20000
    t_bet_samp = 20 * tau_e
    steps_bet_samp = int(t_bet_samp/dt)
    
    print("Working with " + str(N_pat) + " patterns")
      
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
    #   Sample Langevin, Sample!   #
    #------------------------------#
        
      
    print("Start time : ", datetime.datetime.now())
           
    for alpha in range(N_pat):
        print("sampling pattern " + str(alpha))
        mu_ext = np.concatenate((mu_net[alpha],np.zeros(N)))
        Sigma_solve = np.loadtxt(in_location_langevin+"/Sigma_solve_"+str(alpha))
        init = np.random.multivariate_normal(mean = mu_ext, cov = Sigma_solve)
        u0 = init[:N]
        eta0 = init[N:]
        (u_samples,eta_samples,_,_) = mt.Langevin_sample(A[alpha],B,mu_net[alpha],
                                           sample_size,steps_bet_samp,
                                           u0,eta0)
        
        mu_langevin = np.mean(u_samples, axis=0)
        Sigma_langevin = np.cov(u_samples, rowvar=False)
        Sigma_eta_samp = np.cov(eta_samples, rowvar=False)
        
        #np.save(out_location+"/langevin_samples_"+str(alpha),samples)                
        np.savetxt(out_location+"/mu_langevin_"+str(alpha),mu_langevin)
        np.savetxt(out_location+"/sigma_langevin_"+str(alpha),Sigma_langevin)
        np.savetxt(out_location+"/sigma_eta_samp_"+str(alpha),Sigma_eta_samp)
        np.savetxt(out_location+"/std_langevin_"+str(alpha),np.sqrt(np.diag(Sigma_langevin)))   
    
    
    print("End time : ", datetime.datetime.now())
        
    
# Call Main
if __name__ == "__main__":
    main()
