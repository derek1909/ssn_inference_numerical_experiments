# This code computes example activity traces

import numpy as np
import scipy as sp
import methods as mt
from parameters import *
import datetime, os

def smoothen (x_array, width):
    x_array_smooth = np.copy(x_array)
    n_neurons = len(x_array[0])-1
    n_timebins = len(x_array[:,0])-1
    k = 1.0/(2.0*width*width)
    for t in range(1,n_timebins+1):
        kernell = np.exp(-k*(x_array[t,0] - x_array[1:,0])**2)
        kernell /= np.sum(kernell)
        for neuron in range(n_neurons):
            x_array_smooth[t,1+neuron] = np.sum(x_array[1:,1+neuron]*kernell)
    return x_array_smooth

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
        out_location = "results/examples/w_lagged_noise"
        lag_time = tau_e/5.0 # 4ms
        buffer_size =int(lag_time/dt)
    else:
        out_location = "results/examples"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    
    epsilon = 1.0E-12
    
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
        mu_0[alpha] = np.loadtxt(in_location+"/mu_evolved_net_"+str(alpha))
        Sigma_0[alpha] = np.loadtxt(in_location+"/sigma_evolved_net_"+str(alpha))
                
    total_time = 1 # 1 s
    t_bet_samp = 5e-3 # 5ms
    sample_size = int(total_time/t_bet_samp)
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
        
        r_samples = mt.get_r(u_samples)
        
        example_mat_exc = np.empty([sample_size+1, N_exc+1])
        example_mat_inh = np.empty([sample_size+1, N_inh+1])
        example_mat_exc[0,0] = 0
        example_mat_inh[0,0] = 0
        example_mat_exc[1:,0] = np.linspace(0.0, 1000*total_time, num=sample_size, endpoint=False)
        example_mat_inh[1:,0] = np.linspace(0.0, 1000*total_time, num=sample_size, endpoint=False)
        example_mat_exc[0,1:] = np.linspace(-90, 90, num=N_exc, endpoint=False)
        example_mat_inh[0,1:] = np.linspace(-90, 90, num=N_inh, endpoint=False)
        example_mat_exc[1:,1:] = u_samples[:,:N_exc]
        example_mat_inh[1:,1:] = u_samples[:,N_exc:]
        np.savetxt(out_location+"/example_u_exc_"+str(alpha),example_mat_exc.T)
        np.savetxt(out_location+"/example_u_inh_"+str(alpha),example_mat_inh.T)
        
        width = 20
        
        example_mat_exc_smooth = smoothen (example_mat_exc, width)
        example_mat_inh_smooth = smoothen (example_mat_inh, width)
        
        np.savetxt(out_location+"/example_u_exc_smooth_"+str(alpha),example_mat_exc_smooth.T)
        np.savetxt(out_location+"/example_u_inh_smooth_"+str(alpha),example_mat_inh_smooth.T)
        
                
        example_mat_exc[1:,1:] = r_samples[:,:N_exc]
        example_mat_inh[1:,1:] = r_samples[:,N_exc:]
        np.savetxt(out_location+"/example_r_exc_"+str(alpha),example_mat_exc.T)
        np.savetxt(out_location+"/example_r_inh_"+str(alpha),example_mat_inh.T)
        
        example_mat_exc_smooth = smoothen (example_mat_exc, width)
        example_mat_inh_smooth = smoothen (example_mat_inh, width)
        
        np.savetxt(out_location+"/example_r_exc_smooth_"+str(alpha),example_mat_exc_smooth.T)
        np.savetxt(out_location+"/example_r_inh_smooth_"+str(alpha),example_mat_inh_smooth.T)
        
    print("End time : ", datetime.datetime.now())
     

# Call Main
if __name__ == "__main__":
    main()
