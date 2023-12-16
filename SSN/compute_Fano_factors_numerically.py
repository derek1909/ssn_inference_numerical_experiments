# Code to compute Fano factors numerically for doubly stochastic Poisson and Gamma processes

import numpy as np
import scipy as sp
from scipy.stats import poisson
import methods as mt
from parameters import *
import datetime, os

def Gamma_spike_count_time_resc(shape, T):
    n_steps = max(5,np.ceil(1.5*T).astype(int))
    mean = 1.0
    scale = mean/shape
    
    # we first randomize the start time
    
    start_time = np.random.rand()
    
    spike_times = np.random.gamma(shape, scale, size=n_steps)
    tot_time = np.sum(spike_times)
    
    while (tot_time<start_time):
        new_times = np.random.gamma(shape, scale, size=n_steps)
        tot_time += np.sum(new_times)
        spike_times = np.concatenate((spike_times,new_times))    
    
    done = False
    i = 0
    rest = start_time
    while not done:
        if (spike_times[i] >= rest):
            spike_times[i] -= rest
            done = True
        else:
            rest -= spike_times[i]
            spike_times[i] = 0.0
            i += 1
             
    spike_times = spike_times[i:]
    
    # now we make sure we have a long enough train
     
    tot_time = np.sum(spike_times)
    
    while (tot_time < T):
        new_times = np.random.gamma(shape, scale, size=n_steps)
        tot_time += np.sum(new_times)
        spike_times = np.concatenate((spike_times,new_times))
    
    time = 0.0
    n_spikes = 0
    while time < T:
        time += spike_times[n_spikes]
        if (not time > T):
            n_spikes += 1
        
    return n_spikes
    
    
def produce_Fano_factors_1_input(alpha, h, mu_0, Sigma_0, Sigma_eta, W,
                                    n_win, n_win_per_batch, 
                                    steps_win, t_win_fano,
                                    sample_size_per_batch, n_batches,
                                    time_resc, Poisson, shape,
                                    steps_bet_samp, out_location):
                                    
    spike_counts = np.empty([n_win,N], dtype=int)
    u0 = np.random.multivariate_normal(mean = mu_0[alpha], cov = Sigma_0[alpha])
    eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
    print("evolving pattern" + str(alpha))
    u, eta = mt.network_evolution(W,h[alpha],u0,Sigma_eta, eta = eta0)
    print("sampling from the network")
    
    mean_rate = np.zeros(N)
    
    for batch in range(n_batches):
        print("pattern ", alpha, ", batch ", batch)
    
        (u_samples,eta_samples,_,_) = mt.network_sample(W,h[alpha],u,eta,sample_size_per_batch,
                                            steps_bet_samp,Sigma_eta)
    
        r_samples = mt.get_r(u_samples)
    
        selected_r_samples = np.reshape(r_samples,(2*n_win_per_batch,steps_win,N))[::2,:,:]
        
        
        start = batch*n_win_per_batch
        stop = (batch+1)*n_win_per_batch
        
        if (time_resc == True):
            rate_integrals = dt*np.sum(selected_r_samples, axis =1)
            for win in range(n_win_per_batch):
                for i in range(N):
                    spike_counts[start+win,i] = Gamma_spike_count_time_resc(shape,
                                                            rate_integrals[win,i])                
        else:
            spike_counts[start:stop] = np.sum(poisson.rvs(dt*selected_r_samples),axis=1)
    
    variance = np.var(spike_counts, axis = 0)
    mean = np.mean(spike_counts, axis = 0)
    Fano_factors = variance/mean
    
    mean_rate = (1.0*mean)/t_win_fano
    
    if (time_resc == True):
        if Poisson:
            np.savetxt(out_location+"/fano_evolved_net_Poisson_time_resc_"+str(alpha),Fano_factors)
            np.savetxt(out_location+"/check_mean_rate_Poisson_time_resc_"+str(alpha),mean_rate)
        else:
            np.savetxt(out_location+"/fano_evolved_net_Gamma_form_"+str(shape)+"time_resc_"+str(alpha),Fano_factors)
            np.savetxt(out_location+"/check_mean_rate_Gamma_form_"+str(shape)+"time_resc_"+str(alpha),mean_rate)
    else:
        np.savetxt(out_location+"/fano_evolved_net_Poisson_direct_"+str(alpha),Fano_factors)
        np.savetxt(out_location+"/check_mean_rate_Poisson_direct_"+str(alpha),mean_rate)
        
        
############
#   Main   #
############

def main():
    
    in_location = "parameter_files"
    
    custom_contrasts = False
    
    time_resc = True
    Poisson = False
    
    if time_resc:
        print("Using time rescaling theorem")
        if Poisson:
            shape = 1.0
            print("Poisson distribution")
        else:
            shape = 1.15 #2.0
            print("Gamma distribution. shape = ", shape)
        
    if (custom_contrasts == True):
        out_location = in_location+"/Fano_factors_custom_contrasts"
    else:
        out_location = in_location+"/Fano_factors"
    if not os.path.exists(out_location):
        os.makedirs(out_location)  
        
    epsilon = 1.0E-12
    
    t_win_fano = 100.0e-3 # time window for Fano factor from the Ponce paper
    n_win_per_batch = 1000 # number of windows per batch for Fano factor calculation
    n_batches = 500
    
    n_win = n_batches * n_win_per_batch
    
    t_bet_samp = dt
    steps_bet_samp = 1  
    
    total_time_per_batch = 2 * t_win_fano *  n_win_per_batch # we will discard every other window
    
    sample_size_per_batch = int(total_time_per_batch/t_bet_samp)
    steps_win = int(t_win_fano/t_bet_samp)
        
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
            
    print("Start time : ", datetime.datetime.now())
    
    r_samples = np.empty([sample_size_per_batch,N])
    
    for alpha in range(N_pat):
        produce_Fano_factors_1_input(alpha, h, mu_0, Sigma_0, Sigma_eta, W,
                                    n_win, n_win_per_batch, steps_win, t_win_fano,
                                    sample_size_per_batch, n_batches,
                                    time_resc, Poisson, shape,
                                    steps_bet_samp, out_location)
                
    print("End time : ", datetime.datetime.now())
        

# Call Main
if __name__ == "__main__":
    main()
