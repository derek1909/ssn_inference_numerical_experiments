import numpy as np
import scipy as sp
import methods as mt
from parameters import *
import datetime, os, sys

############
#   Main   #
############

def main():

    from_trainset = False
    
    if (from_trainset):
        final_contrast_level = 3
    else:
        final_contrast_level = 10
        contrast = 0.35        
        #contrast = 0.7
        #final_contrast_level = "custom"
    
    with_delay = True
    
    in_location = "parameter_files"
    out_location = "results/transient_short"
    out_location_indiv = "results/individual_neurons_0_"+str(final_contrast_level)
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    if not os.path.exists(out_location_indiv):
        os.makedirs(out_location_indiv)
    
    #stats = 10000
    #stats = 100
    stats = 20
               
    #t_init = 200.0e-3 
    #t_stimulus = 500.0e-3 
    #t_final = 100.0e-3 
    t_init = 225.0e-3 
    t_stimulus = 100.0e-3 
    t_final = 125.0e-3 
    
    total_time = t_init + t_stimulus + t_final
    t_bet_samp = 2*dt
    #t_bet_samp = 5*dt
    steps_bet_samp = int(t_bet_samp/dt)    
    
    if (with_delay):
        mean_delay = 45.0e-3
        delay_sd = 5.0e-3
        mean_delay_points = int(mean_delay/t_bet_samp)
        print("Using random delay times")
        delays = np.random.normal(mean_delay, delay_sd, N_exc)
        delays[delays<0] = 0.0
        delays = np.concatenate((delays,delays))        
    else:
        delays = np.zeros(N)   
        mean_delay_points = 0
    
             
    samp_rec = min(20,stats)
    
    points_init = int(t_init/t_bet_samp)
    points_stimulus = int(t_stimulus/t_bet_samp)
    points_final = int(t_final/t_bet_samp)
    
    sample_size = points_init + points_stimulus + points_final
                     
    T_window = 100.0e-3 # 100 ms
    points_per_window = int(T_window/t_bet_samp)
        
    times = points_per_window//2
    freqs = points_per_window//2 + 1
    
    epsilon = 1.0E-12
    
    bin_size = 10.0e-3
    steps_bin =  int(bin_size/t_bet_samp)
    binned_sample_size =  int(sample_size/steps_bin)
      
    #----------------------------------------------#
    #   We import W, h, and the noise covariance   #
    #----------------------------------------------#

    Sigma_eta = np.loadtxt(in_location+"/sigma_eta_learn")
    h_0 = np.loadtxt(in_location+"/h_true_0_learn")
    mu_0 = np.loadtxt(in_location+"/mu_evolved_net_0")
    Sigma_0= np.loadtxt(in_location+"/sigma_evolved_net_0")
        
    if (from_trainset):
        h_final = np.loadtxt(in_location+"/h_true_"+str(final_contrast_level)+"_learn")
        mu_final = np.loadtxt(in_location+"/mu_evolved_net_"+str(final_contrast_level))
        Sigma_final= np.loadtxt(in_location+"/sigma_evolved_net_"+str(final_contrast_level))
    else:
        input_scaling_learn = np.loadtxt(in_location+"/input_scaling_learn")
        input_baseline_learn = np.loadtxt(in_location+"/input_baseline_learn")
        input_nl_pow_learn = np.loadtxt(in_location+"/input_nl_pow_learn")
        h_full_contrast = np.loadtxt(in_location+"/h4")
        h_lin= contrast * h_full_contrast
        h_final = input_scaling_learn * np.power((h_lin + input_baseline_learn),input_nl_pow_learn)
        
        mu_final = np.loadtxt(in_location+"/mu_evolved_net_4")
        Sigma_final= np.loadtxt(in_location+"/sigma_evolved_net_4")
    
    W = np.loadtxt(in_location+"/w_learn")
       
    #------------------------------#
    #   Evolution of the moments   #
    #------------------------------#
           
    # We take simple initial conditions for mu and Sigma
            
    print("Start time : ", datetime.datetime.now())
    
    u_samples = np.empty([stats,sample_size,N])    
    eta_samples = np.empty([stats,sample_size,N])  
    r_samples = np.empty([stats,sample_size,N])
    
    for s in range(stats):
        print("s = "+str(s))
        while True:
            try:
                u0 = np.random.multivariate_normal(mean = mu_0, cov = Sigma_0)
                eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
                # We take a small number of new steps to make sure the network is in equilibrium
                u0, eta0 = mt.network_evolution(W,h_0,u0,Sigma_eta,100000,eta = eta0)
                done = 0
                
                # We first collect samples from the spontaneous activity
                (u_samples[s,done:(done+points_init)],
                eta_samples[s,done:(done+points_init)],u,eta) =\
                                     mt.network_sample(W,h_0,
                                    u0,eta0,points_init,steps_bet_samp,Sigma_eta)
                done = points_init
                # jumping to higher input level while sampling from the network
                (u_samples[s,done:(done+points_stimulus)],eta_samples[s,done:(done+points_stimulus)],u,eta) = mt.network_sample_w_delay(W,h_0,
                                    h_final,delays,
                                    u,eta,points_stimulus,steps_bet_samp,Sigma_eta)
                done += points_stimulus
                # we finally go back to spontaneous activity
                (u_samples[s,done:],eta_samples[s,done:],u,eta) = mt.network_sample_w_delay(W,
                                    h_final,h_0,delays,
                                    u,eta,points_final,steps_bet_samp,Sigma_eta)
                
            except:
                print("######## ERROR ######## : \n", sys.exc_info()[0])
                continue
            
            break
                       
        r_samples[s] = mt.get_r(u_samples[s])
            
    u_indiv = np.empty([sample_size,samp_rec + 3])
    r_indiv = np.empty([sample_size,samp_rec + 3])
    
    u_mean = np.empty([sample_size,N+1])    
    r_mean = np.empty([sample_size,N+1])
    
    u_std = np.empty([sample_size,N+1])    
    r_std = np.empty([sample_size,N+1])
    
    u_pop_mean = np.empty([sample_size,5])    
    r_pop_mean = np.empty([sample_size,5])
    
    u_mean[:,0] = t_bet_samp*np.arange(sample_size)
    r_mean[:,0] = t_bet_samp*np.arange(sample_size)
    
    u_std[:,0] = t_bet_samp*np.arange(sample_size)
    r_std[:,0] = t_bet_samp*np.arange(sample_size)
    
    u_pop_mean[:,0] = t_bet_samp*np.arange(sample_size)
    r_pop_mean[:,0] = t_bet_samp*np.arange(sample_size)
    
    u_mean[:,1:] = np.average(u_samples, axis = 0)
    r_mean[:,1:] = np.average(r_samples, axis = 0)
    
    u_std[:,1:] = np.std(u_samples, axis = 0)
    r_std[:,1:] = np.std(r_samples, axis = 0)
    
    u_pop_mean[:,1] = np.average(u_mean[:,1:(N_exc+1)],axis=1)
    u_pop_mean[:,2] = np.std(u_mean[:,1:(N_exc+1)],axis=1)
    u_pop_mean[:,3] = np.average(u_mean[:,(N_exc+1):(N+1)],axis=1)
    u_pop_mean[:,4] = np.std(u_mean[:,(N_exc+1):(N+1)],axis=1)
    
    r_pop_mean[:,1] = np.average(r_mean[:,1:(N_exc+1)],axis=1)
    r_pop_mean[:,2] = np.std(r_mean[:,1:(N_exc+1)],axis=1)
    r_pop_mean[:,3] = np.average(r_mean[:,(N_exc+1):(N+1)],axis=1)
    r_pop_mean[:,4] = np.std(r_mean[:,(N_exc+1):(N+1)],axis=1)
    
    u_pop_mean_binned = np.empty([binned_sample_size,3])    
    r_pop_mean_binned = np.empty([binned_sample_size,3])
    
    for t in range(binned_sample_size):
        u_pop_mean_binned[t,0] = t*steps_bin*t_bet_samp
        r_pop_mean_binned[t,0] = t*steps_bin*t_bet_samp
        
        u_pop_mean_binned[t,1] = np.average(u_pop_mean[t*steps_bin:(t+1)*steps_bin,1])
        u_pop_mean_binned[t,2] = np.average(u_pop_mean[t*steps_bin:(t+1)*steps_bin,3])
        r_pop_mean_binned[t,1] = np.average(r_pop_mean[t*steps_bin:(t+1)*steps_bin,1])
        r_pop_mean_binned[t,2] = np.average(r_pop_mean[t*steps_bin:(t+1)*steps_bin,3])
        
    for i in range(N):
        u_indiv = np.column_stack((u_mean[:,0],u_mean[:,1+i],u_std[:,1+i],u_samples[:samp_rec,:,i].T))
        r_indiv = np.column_stack((r_mean[:,0],r_mean[:,1+i],r_std[:,1+i],r_samples[:samp_rec,:,i].T))
        np.savetxt(out_location_indiv+"/u_evolution_"+str(i), u_indiv)
        np.savetxt(out_location_indiv+"/r_evolution_"+str(i), r_indiv)    
    
    np.savetxt(out_location+"/delays_transient_0_"+str(final_contrast_level),delays)
    
    np.savetxt(out_location+"/u_means_transient_0_"+str(final_contrast_level),u_mean)
    np.savetxt(out_location+"/r_means_transient_0_"+str(final_contrast_level),r_mean)
    
    np.savetxt(out_location+"/u_stds_transient_0_"+str(final_contrast_level),u_std)
    np.savetxt(out_location+"/r_stds_transient_0_"+str(final_contrast_level),r_std)
               
    np.savetxt(out_location+"/u_pop_mean_transient_0_"+str(final_contrast_level),u_pop_mean)
    np.savetxt(out_location+"/r_pop_mean_transient_0_"+str(final_contrast_level),r_pop_mean)
    
    np.savetxt(out_location+"/u_pop_mean_binned_transient_0_"+str(final_contrast_level),u_pop_mean_binned)
    np.savetxt(out_location+"/r_pop_mean_binned_transient_0_"+str(final_contrast_level),r_pop_mean_binned)
        
    np.save(out_location+"/u_samples_transient_0_"+str(final_contrast_level),u_samples[:samp_rec])
    np.save(out_location+"/r_samples_transient_0_"+str(final_contrast_level),r_samples[:samp_rec])
    np.save(out_location+"/eta_samples_transient_0_"+str(final_contrast_level),eta_samples[:samp_rec])
    
    
    print("End time : ", datetime.datetime.now())
        

# Call Main
if __name__ == "__main__":
    main()
