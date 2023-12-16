# This code takes a weight matrix and a set of inputs and computes the stationary
# solution for the output moments using the full network simulation

import numpy as np
import scipy as sp
import methods as mt
from parameters import *

import datetime, os

############
#   Main   #
############

def main():

    lagged_noise = False
    in_location = "parameter_files"
    
    if (lagged_noise == True):
        print("########################")
        print("#  USING LAGGED NOISE  #")
        print("########################")
        out_location = in_location+"/w_lagged_noise"
        lag_time = tau_e/5.0 # 4ms
        buffer_size =int(lag_time/dt)
    else:
        out_location = "results"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    
      
    print("Working with " + str(N_pat) + " patterns")
      
    #----------------------------------------------#
    #   We import W, h, and the noise covariance   #
    #----------------------------------------------#

    Sigma_eta = np.loadtxt(in_location+"/sigma_eta_learn")
      
    h = np.empty([N_pat,N])
    
    for alpha in range(N_pat):
        h[alpha] = np.loadtxt(in_location+"/h_true_"+str(alpha)+"_learn")
        
    W = np.loadtxt(in_location+"/w_learn")
       
    #------------------------------#
    #   Evolution of the network   #
    #------------------------------#
           
    mu_0 = np.empty([N_pat,N])
    Sigma_0 = np.empty([N_pat,N,N])
           
    for alpha in range(N_pat):
        mu_0[alpha] = np.loadtxt(in_location+"/mu_learn_"+str(alpha))
        Sigma_0[alpha] = np.loadtxt(in_location+"/sigma_learn_"+str(alpha))
        Sigma_0[alpha] = regularize_Sigma(Sigma_0[alpha], out_location, "Sigma_0"+str(alpha),epsilon)
        
    mu_final = np.empty([N_pat,N])
    Sigma_final = np.empty([N_pat,N,N])
    
    nu_final = np.empty([N_pat,N])
    Lambda_final = np.empty([N_pat,N,N])
    
    sample_size = 20000
    t_bet_samp = 10 * tau_e
    steps_bet_samp = int(t_bet_samp/dt)  
    
    
    print("Start time : ", datetime.datetime.now())
    
    if (lagged_noise == True):
        L = np.linalg.cholesky(Sigma_eta)
        
    for alpha in range(N_pat):
        u0 = np.random.multivariate_normal(mean = mu_0[alpha], cov = Sigma_0[alpha])
        eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
        if (lagged_noise == True):
            print("Buffering noise")
            buffered_noise = mt.buffer_init(buffer_size,L)
            
            print("evolving pattern" + str(alpha))
            (u,eta,buffered_noise) = mt.network_evolution_w_lagged_noise(W,h[alpha],
                            u0,Sigma_eta,L,buffered_noise, eta = eta0)
            
            print("sampling from the network")
            (u_samples,eta_samples,_,_,buffered_noise) = mt.network_sample_w_lagged_noise(W,h[alpha],
                                        u,eta,sample_size,steps_bet_samp,Sigma_eta,L,buffered_noise)
            
            np.save(out_location+"/eta_samples_"+str(alpha), eta_samples)
        
        else:
            print("evolving pattern " + str(alpha))
            u, eta = mt.network_evolution(W,h[alpha],u0,Sigma_eta, eta = eta0)
            print("sampling moments")
            (u_samples,_,_,_) = mt.network_sample(W,h[alpha],u,eta,sample_size,steps_bet_samp,Sigma_eta)
        
        mu_final[alpha] = np.mean(u_samples, axis=0)
        Sigma_final[alpha] = np.cov(u_samples, rowvar=False)
        r_samples = mt.get_r(u_samples)
        nu_final[alpha] = np.mean(r_samples, axis=0)
        Lambda_final[alpha] = np.cov(r_samples, rowvar=False)
        
        np.savetxt(out_location+"/mu_evolved_net_"+str(alpha),mu_final[alpha])
        np.savetxt(out_location+"/sigma_evolved_net_"+str(alpha),Sigma_final[alpha])
        np.savetxt(out_location+"/std_evolved_net_"+str(alpha),np.sqrt(np.diag(Sigma_final[alpha])))
                
        np.savetxt(out_location+"/nu_evolved_net_"+str(alpha),nu_final[alpha])
        np.savetxt(out_location+"/lambda_evolved_net_"+str(alpha),Lambda_final[alpha])
        np.savetxt(out_location+"/std_r_evolved_net_"+str(alpha),np.sqrt(np.diag(Lambda_final[alpha])))
                                                                  
        np.savetxt(out_location+"/all_mus_evolved_net",mu_final)
        np.savetxt(out_location+"/all_nus_evolved_net",nu_final)
        
    print("End time : ", datetime.datetime.now())
        
        
    

# Call Main
if __name__ == "__main__":
    main()
