import numpy as np
import scipy as sp
from scipy.stats import circstd
from scipy import signal
import methods as mt
from parameters import *
import datetime, os, sys

############
#   Main   #
############

def main():
    
    with_delay = True
        
    custom_contrasts = False
    
    if custom_contrasts:
        contrast = 0.7
        final_contrast_level = "custom"
    else:        
        final_contrast_level = 3
    
    remove_mean = True

    in_location = "parameter_files"      
    out_location = "results/transient/phase_reset"
    out_location_tuning = "results/transient/tuning"

    if (with_delay): 
        out_location = out_location + "/with_delays"
        out_location_tuning = out_location_tuning + "/with_delays"
    
    if (remove_mean):
        out_location = out_location + "/remove_mean"
        out_location_tuning = out_location_tuning + "/remove_mean"

    out_location_indiv = out_location_tuning+"/individual_neurons_0_"+str(final_contrast_level)
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
        
    if not os.path.exists(out_location_tuning):
        os.makedirs(out_location_tuning)
    
    if not os.path.exists(out_location_indiv):
        os.makedirs(out_location_indiv)
   
        
    n_trials = 100
    n_delay_configs = 1000
    
    samp_rec = min (n_trials,20)
            
    t_init = 400.0e-3 # 100ms
    t_stimulus = 400.0e-3 #400ms
    t_final = t_init
    
    lowcut = 20.0
    highcut = 80.0
    
    total_time = t_init + t_stimulus + t_final
    t_bet_samp = 2*dt
    steps_bet_samp = int(t_bet_samp/dt)    
    
    if (with_delay):
        mean_delay = 45.0e-3
        delay_sd = 5.0e-3
        print("Using random delay times")
        delays = np.random.normal(mean_delay, delay_sd, N_exc)
        delays[delays<0] = 0.0
        delays = np.concatenate((delays,delays))
    else:
        delays = np.zeros(N)    
    
    np.savetxt(out_location+"/delays_transient_0_"+str(final_contrast_level),delays)    

    points_init = int(t_init/t_bet_samp)
    points_stimulus = int(t_stimulus/t_bet_samp)
    points_final = int(t_final/t_bet_samp)
    
    sample_size = points_init + points_stimulus + points_final

    T_window = 100.0e-3 # 100 ms
    points_per_window = int(T_window/t_bet_samp)
        
    times = points_per_window//2
    freqs = points_per_window//2 + 1
    
    ts = t_bet_samp
    
    epsilon = 1.0E-12
    
    bin_size = 10.0e-3
    steps_bin =  int(bin_size/t_bet_samp)
    binned_sample_size =  int(sample_size/steps_bin)
      
    #----------------------------------------------#
    #   We import W, h, and the noise covariance   #
    #----------------------------------------------#

    Sigma_eta = np.loadtxt(in_location+"/sigma_eta_learn")
    Sigma_eta = regularize_Sigma(Sigma_eta, out_location, "sigma_eta_learn",epsilon)
     
    h_0 = np.loadtxt(in_location+"/h_true_0_learn")
    mu_0 = np.loadtxt(in_location+"/mu_evolved_net_0")
    nu_0 = np.loadtxt(in_location+"/nu_evolved_net_0")
    Sigma_0= np.loadtxt(in_location+"/sigma_evolved_net_0")
    Sigma_0 = regularize_Sigma(Sigma_0, out_location, "Sigma_0",epsilon)
    
    W = np.loadtxt(in_location+"/w_learn")
    
    if (custom_contrasts):
        input_scaling_learn = np.loadtxt(in_location+"/input_scaling_learn")
        input_baseline_learn = np.loadtxt(in_location+"/input_baseline_learn")
        input_nl_pow_learn = np.loadtxt(in_location+"/input_nl_pow_learn")
        h_full_contrast = np.loadtxt(in_location+"/h4")
        h_final = input_scaling_learn * np.power((contrast * h_full_contrast + input_baseline_learn),
                                                    input_nl_pow_learn)
        
        # obtaining final moments
        sample_size_moments = 20000
        t_bet_samp_moments = 10 * tau_e
        steps_bet_samp_moments = int(t_bet_samp_moments/dt)
        
        u0 = np.random.multivariate_normal(mean = mu_0, cov = Sigma_0)
        eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
        
        u, eta = mt.network_evolution(W,h_final,u0,Sigma_eta, eta = eta0)
        print("sampling to obtain moments")
        (u_samples,_,_,_) = mt.network_sample(W,h_final,u,eta,
                            sample_size_moments,steps_bet_samp_moments,Sigma_eta)    
        print("computing moments")
        mu_final = np.mean(u_samples, axis=0)
        Sigma_final = np.cov(u_samples, rowvar=False)
        r_samples = mt.get_r(u_samples)
        nu_final= np.mean(r_samples, axis=0)
        np.savetxt(out_location_tuning +"/mu_evolved_net_"+str(final_contrast_level), mu_final)
        np.savetxt(out_location_tuning +"/nu_evolved_net_"+str(final_contrast_level), nu_final)
        np.savetxt(out_location_tuning +"/sigma_evolved_net_"+str(final_contrast_level), Sigma_final)
        
    else:
        h_final = np.loadtxt(in_location+"/h_true_"+str(final_contrast_level)+"_learn")
        mu_final = np.loadtxt(in_location+"/mu_evolved_net_"+str(final_contrast_level))
        nu_final = np.loadtxt(in_location+"/nu_evolved_net_"+str(final_contrast_level))
        Sigma_final = np.loadtxt(in_location+"/sigma_evolved_net_"+str(final_contrast_level))
    
    Sigma_ = Sigma_final[:N_exc,:N_exc]
    eivals, eivecs = np.linalg.eigh(Sigma_)
                
    idx = eivals.argsort()[::-1] 
    eigenvalues = eivals[idx]
    eigenvectors = eivecs[:,idx]
    
    u_samples = np.empty([n_trials,sample_size,N])
    eta_samples = np.empty([n_trials,sample_size,N])  
    r_samples = np.empty([n_trials,sample_size,N])
    u_mean = np.empty([sample_size,N+1])    
    r_mean = np.empty([sample_size,N+1])
    u_std = np.empty([sample_size,N+1])    
    r_std = np.empty([sample_size,N+1])
        
    u_mean[:,0] = ts * np.arange(0,sample_size)
    r_mean[:,0] = ts * np.arange(0,sample_size)
    u_std[:,0] = ts * np.arange(0,sample_size)
    r_std[:,0] = ts * np.arange(0,sample_size)
    
    u_mean_all_delays = np.empty([n_delay_configs,sample_size,N+1])    
    r_mean_all_delays = np.empty([n_delay_configs,sample_size,N+1])
        
    phases_u = np.empty([n_trials, sample_size])
    phases_r = np.empty([n_trials, sample_size])
    
    hilberts_u = np.empty([n_trials, sample_size])
    hilberts_r = np.empty([n_trials, sample_size])
    
    check_phase_u = np.empty([sample_size,4])
    check_phase_r = np.empty([sample_size,4])
    phases_u_SD = np.empty([sample_size,2])
    phases_r_SD = np.empty([sample_size,2])
    
    phases_u_SD[:,0] = ts * np.arange(0,sample_size)
    phases_r_SD[:,0] = ts * np.arange(0,sample_size)
    check_phase_u[:,0] = ts * np.arange(0,sample_size)
    check_phase_r[:,0] = ts * np.arange(0,sample_size)
    
    phases_u_SD_av = np.zeros((sample_size,2))
    phases_r_SD_av = np.zeros((sample_size,2))
    
    overshoot_u =  np.empty([N])
    overshoot_r =  np.empty([N])
    undershoot_u =  np.empty([N])
    undershoot_r =  np.empty([N])
    
    overshoot_u_av =  np.zeros(N)
    overshoot_r_av =  np.zeros(N)
    undershoot_u_av =  np.zeros(N)
    undershoot_r_av =  np.zeros(N)
    
    
    #defining parameters for bandpass filtering
    fs = 1.0/ts
    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b_filter, a_filter = signal.butter(4, [low,high], 'band')
    
    w_filter, h_filter = signal.freqz(b_filter, a_filter, worN=2000)
    filter_plot = np.empty([len(h_filter),2])
    filter_plot[:,0] = (fs * 0.5 / np.pi) * w_filter
    filter_plot[:,1] = np.abs(h_filter)
    np.savetxt(out_location+"/bandpass_filter",filter_plot)
    
    
    #--------------#
    #   Sampling   #
    #--------------#
           
    print("Start time : ", datetime.datetime.now())
    
    for dconf in range(n_delay_configs):
        print("delay configuration number "+str(dconf))
        
        if (with_delay):
            delays = np.random.normal(mean_delay, delay_sd, N_exc)
            delays[delays<0] = 0.0
            delays = np.concatenate((delays,delays))
            np.savetxt(out_location+"/delays_transient_0_"+str(final_contrast_level)
                                                        +"_dconf_"+str(dconf),delays)
        for s in range(n_trials):
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
        
        r_samples = mt.get_r(u_samples)
                           
        uE_pop_mean = np.average(u_samples[:,:,:N_exc], axis = 2)
        rE_pop_mean = np.average(r_samples[:,:,:N_exc], axis = 2)
        
        uE_pop_mean_filtered = np.copy(uE_pop_mean)
        rE_pop_mean_filtered = np.copy(rE_pop_mean)   
        
        if (remove_mean):
            uE_pop_mean_filtered -= np.mean(uE_pop_mean_filtered, axis = 0)
            rE_pop_mean_filtered -= np.mean(rE_pop_mean_filtered, axis = 0)

        for s in range(n_trials):
            uE_pop_mean_filtered[s] = signal.lfilter(b_filter, a_filter, 
                                        uE_pop_mean[s]) # we bandpass filter the signal
            rE_pop_mean_filtered[s] = signal.lfilter(b_filter, a_filter, 
                                        rE_pop_mean[s]) # we bandpass filter the signal
            
            uE_pop_mean_analytic = signal.hilbert(uE_pop_mean_filtered[s])
            rE_pop_mean_analytic = signal.hilbert(rE_pop_mean_filtered[s])
            hilberts_u[s] = np.imag(uE_pop_mean_analytic)
            hilberts_r[s] = np.imag(rE_pop_mean_analytic)
            phases_u[s] = np.angle(uE_pop_mean_analytic)
            phases_r[s] = np.angle(rE_pop_mean_analytic) 
        
        check_phase_u[:,1] = uE_pop_mean[s]    
        check_phase_u[:,2] = uE_pop_mean_filtered[s]
        check_phase_u[:,3] = hilberts_u[s]
        check_phase_r[:,1] = rE_pop_mean[s]
        check_phase_r[:,2] = rE_pop_mean_filtered[s]
        check_phase_r[:,3] = hilberts_r[s]

        phases_u_SD[:,1] = circstd(phases_u, axis = 0)
        phases_r_SD[:,1] = circstd(phases_r, axis = 0)
        
        phases_u_SD_av += phases_u_SD
        phases_r_SD_av += phases_r_SD
            
        
        u_mean[:,1:] = np.average(u_samples,axis = 0)
        r_mean[:,1:] = np.average(r_samples,axis = 0)
        u_std[:,1:] = np.std(u_samples,axis = 0)
        r_std[:,1:] = np.std(r_samples,axis = 0)
        
        u_mean_all_delays[dconf] = np.copy(u_mean)
        r_mean_all_delays[dconf] = np.copy(r_mean)     
                                
        overshoot_u  = np.amax(u_mean[:,1:],axis = 0) - mu_final
        overshoot_r  = np.amax(r_mean[:,1:],axis = 0) - nu_final
        undershoot_u  = np.amin(u_mean[:,1:],axis = 0) - mu_0
        undershoot_r  = np.amin(r_mean[:,1:],axis = 0) - nu_0  
        
        overshoot_u_av += overshoot_u
        overshoot_r_av += overshoot_r
        undershoot_u_av += undershoot_u
        undershoot_r_av += undershoot_r
        
        if (dconf < 100):
            np.savetxt(out_location+"/check_phase_u_transient_0_"+str(final_contrast_level)
                                                                 +"_dconf_"+str(dconf),check_phase_u)
            np.savetxt(out_location+"/check_phase_r_transient_0_"+str(final_contrast_level)
                                                                 +"_dconf_"+str(dconf),check_phase_r)
            
            np.savetxt(out_location+"/phases_u_transient_0_"+str(final_contrast_level)
                                                            +"_dconf_"+str(dconf),phases_u[0:10])
            np.savetxt(out_location+"/phases_u_SD_transient_0_"+str(final_contrast_level)
                                                                +"_dconf_"+str(dconf),phases_u_SD)
            np.savetxt(out_location+"/phases_r_transient_0_"+str(final_contrast_level)
                                                            +"_dconf_"+str(dconf),phases_r[0:10])
            np.savetxt(out_location+"/phases_r_SD_transient_0_"+str(final_contrast_level)
                                                            +"_dconf_"+str(dconf),phases_r_SD)
            
            np.savetxt(out_location_tuning+"/u_means_transient_0_"+str(final_contrast_level)
                                                                +"_dconf_"+str(dconf),u_mean)
            np.savetxt(out_location_tuning+"/r_means_transient_0_"+str(final_contrast_level)
                                                                +"_dconf_"+str(dconf),r_mean)
            
            np.savetxt(out_location_tuning+"/overshoot_u_transient_0_"+str(final_contrast_level)
                                                                +"_dconf_"+str(dconf),overshoot_u)
            np.savetxt(out_location_tuning+"/overshoot_r_transient_0_"+str(final_contrast_level)
                                                                +"_dconf_"+str(dconf),overshoot_r)
            
            np.savetxt(out_location_tuning+"/undershoot_u_transient_0_"+str(final_contrast_level)
                                                                +"_dconf_"+str(dconf),undershoot_u)
            np.savetxt(out_location_tuning+"/undershoot_r_transient_0_"+str(final_contrast_level)
                                                                +"_dconf_"+str(dconf),undershoot_r)

        if (dconf < 10):
            u_indiv = np.empty([sample_size,samp_rec + 3])
            r_indiv = np.empty([sample_size,samp_rec + 3])
            u_indiv[:,0] = ts * np.arange(0,sample_size)
            r_indiv[:,0] = ts * np.arange(0,sample_size)
            for i in range(N):
                u_indiv[:,1] = u_mean[:,i+1]
                r_indiv[:,1] = r_mean[:,i+1]
                u_indiv[:,2] = u_std[:,i+1]
                r_indiv[:,2] = r_std[:,i+1]
                u_indiv[:,3:] = u_samples[:samp_rec,:,i].T
                r_indiv[:,3:] = r_samples[:samp_rec,:,i].T
                
                np.savetxt(out_location_indiv+"/conf_"+str(dconf)+"_u_evolution_"+str(i), u_indiv)
                np.savetxt(out_location_indiv+"/conf_"+str(dconf)+"_r_evolution_"+str(i), r_indiv)
    
    
    np.save(out_location_tuning+"/u_means_transient_0_"+str(final_contrast_level)
                                                                +"_all_delay_configs",u_mean_all_delays)

    np.save(out_location_tuning+"/r_means_transient_0_"+str(final_contrast_level)
                                                                +"_all_delay_configs",r_mean_all_delays)
                                                                
    np.save(out_location_tuning+"/u_means_transient_0_"+str(final_contrast_level)
                                                                +"_av_ax_configs",np.average(u_mean_all_delays,axis=0))
    np.save(out_location_tuning+"/r_means_transient_0_"+str(final_contrast_level)
                                                                +"_av_ax_configs",np.average(r_mean_all_delays,axis=0))
                                                                
    np.save(out_location_tuning+"/u_means_transient_0_"+str(final_contrast_level)
                                                                +"_std_ax_configs",np.std(u_mean_all_delays,axis=0))
    np.save(out_location_tuning+"/r_means_transient_0_"+str(final_contrast_level)
                                                                +"_std_ax_configs",np.std(r_mean_all_delays,axis=0))
    
    # Finishing and saving averages across delay configurations
    
    phases_u_SD_av /= (1.0*n_delay_configs)
    phases_r_SD_av /= (1.0*n_delay_configs)
    
    overshoot_u_av /= (1.0*n_delay_configs)
    overshoot_r_av /= (1.0*n_delay_configs)
    undershoot_u_av /= (1.0*n_delay_configs)
    undershoot_r_av /= (1.0*n_delay_configs)
    
    np.savetxt(out_location+"/phases_u_SD_transient_0_"+str(final_contrast_level),phases_u_SD_av)
    np.savetxt(out_location+"/phases_r_SD_transient_0_"+str(final_contrast_level),phases_r_SD_av)
    
    np.savetxt(out_location_tuning+"/overshoot_u_transient_0_"+str(final_contrast_level),overshoot_u_av)
    np.savetxt(out_location_tuning+"/overshoot_r_transient_0_"+str(final_contrast_level),overshoot_r_av)
    
    np.savetxt(out_location_tuning+"/undershoot_u_transient_0_"+str(final_contrast_level),undershoot_u_av)
    np.savetxt(out_location_tuning+"/undershoot_r_transient_0_"+str(final_contrast_level),undershoot_r_av)
    
    
    print("End time : ", datetime.datetime.now())
    

# Call Main
if __name__ == "__main__":
    main()
