# This code computes the autocorrelation and power spectum of the network

import numpy as np
import scipy as sp
import methods as mt
from parameters import *
import datetime
    
# Definition of autocorrelation
    
def autocorr(x, mean, std):
    x = (x - mean)/std
    result = np.correlate(x, x, "same")/len(x)
    return (result[result.size//2:])
    
############
#   Main   #
############

def main():
    
    custom_contrasts = False
    net_location = "results"
    
    if (custom_contrasts == True):
        N_pat = 8
        in_location = net_location + "/correlations_and_power_spectrum_custom_contrasts"
    else:
        N_pat = 5
        in_location = net_location + "/correlations_and_power_spectrum"
        
    out_location = in_location
    
    total_time = 50000.0 * tau_e # 1000s
    t_bet_samp = 2*dt # this should be a multiple of dt
    steps_bet_samp = int(t_bet_samp/dt)    
    sample_size = int(total_time/t_bet_samp)
    
    T_window = 100.0 * tau_e # 2s
    points_per_window = int(T_window/t_bet_samp)
    
    N_windows = int(total_time/T_window) 
    
    times = points_per_window//2 
    freqs = points_per_window//2 + 1
    
    spectrum_u = np.zeros([freqs,2])
    spectrum_r = np.zeros([freqs,2])
    spectrum_LFP = np.zeros([freqs,2])
    
    spectrum_u_norm = np.zeros([freqs,2])
    spectrum_r_norm = np.zeros([freqs,2])
    spectrum_LFP_norm = np.zeros([freqs,2])
    
    autocorr_u = np.zeros([times,2])
    autocorr_r = np.zeros([times,2])
    autocorr_LFP = np.zeros([times,2])
    
    autocorr_u_indiv = np.zeros([N,times,2])
    autocorr_r_indiv = np.zeros([N,times,2])
    
    ts = t_bet_samp
    fs = 1.0/T_window
    
    for i in range(times):
        autocorr_u[i,0] = ts * i
        autocorr_r[i,0] = ts * i
        autocorr_LFP[i,0] = ts * i
        for neuron in range(N):
            autocorr_u_indiv[neuron,i,0] = ts * i
            autocorr_r_indiv[neuron,i,0] = ts * i
    
    for i in range(freqs):    
        spectrum_u[i,0] = fs * i
        spectrum_r[i,0] = fs * i
        spectrum_LFP[i,0] = fs * i
        spectrum_u_norm[i,0] = fs * i
        spectrum_r_norm[i,0] = fs * i
        spectrum_LFP_norm[i,0] = fs * i
            
    for alpha in range(N_pat):
        print("Input level ", alpha)
        u_samples = np.load(in_location+"/u_samples_"+str(alpha)+".npy")
        r_samples = np.load(in_location+"/r_samples_"+str(alpha)+".npy")
        
        autocorr_u[:,1] = np.zeros([times])
        autocorr_r[:,1] = np.zeros([times])
        autocorr_LFP[:,1] = np.zeros([times])
        
        spectrum_u[:,1] = np.zeros([freqs])
        spectrum_r[:,1] = np.zeros([freqs])
        spectrum_LFP[:,1] = np.zeros([freqs])
        
        spectrum_u_norm[:,1] = np.zeros([freqs])
        spectrum_r_norm[:,1] = np.zeros([freqs])
        spectrum_LFP_norm[:,1] = np.zeros([freqs])
        
        autocorr_u_indiv[:,:,1] = np.zeros([N,times])
        autocorr_r_indiv[:,:,1] = np.zeros([N,times])
        
        LFP = np.average(u_samples, axis=1)
        
        done = 0
        mean_LFP = np.mean(LFP)
        var_LFP = np.var(LFP)
        std_LFP = np.sqrt(var_LFP)
        
        for win in range(N_windows):
            samp_LFP = LFP[done:done+points_per_window]
            autocorr_LFP[:,1] += autocorr(samp_LFP,mean_LFP,std_LFP)
            spectrum_LFP[:,1] += np.absolute(np.fft.rfft(samp_LFP-mean_LFP))**2.0
            done = done + points_per_window
        
        mean_us = np.mean(u_samples,axis = 0)
        var_us = np.var(u_samples,axis = 0)
        std_us = np.sqrt(var_us)
        
        mean_rs = np.mean(r_samples,axis = 0)
        var_rs = np.var(r_samples,axis = 0)
        std_rs = np.sqrt(var_rs)
        
        for i in range(N):
            done = 0
            for win in range(N_windows):
                samp_u = u_samples[done:done+points_per_window,i]
                samp_r = r_samples[done:done+points_per_window,i]
                autocorr_u_indiv[i,:,1] += autocorr(samp_u,mean_us[i],std_us[i])
                autocorr_r_indiv[i,:,1] += autocorr(samp_r,mean_rs[i],std_rs[i])
                spec_u_win = np.absolute(np.fft.rfft(samp_u-mean_us[i]))**2.0
                spec_r_win = np.absolute(np.fft.rfft(samp_r-mean_rs[i]))**2.0
                
                spectrum_u[:,1] += spec_u_win
                spectrum_r[:,1] += spec_r_win
                
                spec_u_win *= (var_us[i] / (np.sum(spec_u_win)*fs))
                spec_r_win *= (var_rs[i] / (np.sum(spec_r_win)*fs))
                
                spectrum_u_norm[:,1] += spec_u_win
                spectrum_r_norm[:,1] += spec_r_win
                
                done = done + points_per_window
            
            autocorr_u[:,1] += autocorr_u_indiv[i,:,1]
            autocorr_r[:,1] += autocorr_r_indiv[i,:,1]
        
                
        autocorr_u_indiv[:,:,1] /= (1.0*N_windows)
        autocorr_r_indiv[:,:,1] /= (1.0*N_windows)
            
        autocorr_u[:,1] /= (1.0*N*N_windows)
        autocorr_r[:,1] /= (1.0*N*N_windows)
        autocorr_LFP[:,1] /= (1.0*N_windows)
        
        spectrum_u[:,1] /= (1.0*N*N_windows)
        spectrum_r[:,1] /= (1.0*N*N_windows)
        
        spectrum_u_norm[:,1] /= (1.0*N*N_windows)
        spectrum_r_norm[:,1] /= (1.0*N*N_windows)
        
        
        spectrum_LFP[:,1] /= (1.0*N_windows)
        
        spectrum_LFP_norm[:,1] = spectrum_LFP[:,1] * (var_LFP / (np.sum(spectrum_LFP[:,1])*fs))
        
        np.savetxt(out_location+"/autocorr_u_"+str(alpha),autocorr_u) 
        np.savetxt(out_location+"/autocorr_r_"+str(alpha),autocorr_r)
        np.savetxt(out_location+"/autocorr_LFP_"+str(alpha),autocorr_LFP)
        
        for i in range(N):
            np.savetxt(out_location+"/autocorr_u_"+str(alpha)+"_cell_"+str(i) ,autocorr_u_indiv[i]) 
            np.savetxt(out_location+"/autocorr_r_"+str(alpha)+"_cell_"+str(i) ,autocorr_r_indiv[i])
        
        np.savetxt(out_location+"/spectrum_u_"+str(alpha),spectrum_u) 
        np.savetxt(out_location+"/spectrum_r_"+str(alpha),spectrum_r)
        np.savetxt(out_location+"/spectrum_LFP_"+str(alpha),spectrum_LFP)
        
        np.savetxt(out_location+"/spectrum_u_norm_"+str(alpha),spectrum_u_norm) 
        np.savetxt(out_location+"/spectrum_r_norm_"+str(alpha),spectrum_r_norm)
        np.savetxt(out_location+"/spectrum_LFP_norm_"+str(alpha),spectrum_LFP_norm)

# Call Main
if __name__ == "__main__":
    main()
