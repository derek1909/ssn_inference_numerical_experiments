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
    
    in_location = "results/langevin/autocorr"
    out_location = in_location
    
    total_time = 50000.0 * tau_e # 1000s
    t_bet_samp = 2*dt
    steps_bet_samp = int(t_bet_samp/dt)    
    sample_size = int(total_time/t_bet_samp)
    
    T_window = 100.0 * tau_e # 2s
    points_per_window = int(T_window/t_bet_samp)
    
    N_windows = int(total_time/T_window) 
    
    times = points_per_window//2 
    freqs = points_per_window//2 + 1
    
    spectrum_u = np.zeros([freqs,2])
    spectrum_r = np.zeros([freqs,2])
    
    autocorr_u = np.zeros([times,2])
    autocorr_r = np.zeros([times,2])
    
    ts = t_bet_samp
    fs = 1.0/T_window
    
    for i in range(times):
        autocorr_u[i,0] = ts * i
        autocorr_r[i,0] = ts * i
    for i in range(freqs):    
        spectrum_u[i,0] = fs * i
        spectrum_r[i,0] = fs * i
            
    for alpha in range(N_pat):
        u_samples = np.load(in_location+"/u_samples_"+str(alpha)+".npy")
        r_samples = np.load(in_location+"/r_samples_"+str(alpha)+".npy")
        autocorr_u[:,1] = np.zeros([times])
        autocorr_r[:,1] = np.zeros([times])
        spectrum_u[:,1] = np.zeros([freqs])
        spectrum_r[:,1] = np.zeros([freqs])
        for i in range(N):
            done = 0
            mean_u = np.mean(u_samples[:,i])
            std_u = np.std(u_samples[:,i])
            mean_r = np.mean(r_samples[:,i])
            std_r = np.std(r_samples[:,i])
            for win in range(N_windows):
                samp_u = u_samples[done:done+points_per_window,i]
                samp_r = r_samples[done:done+points_per_window,i]
                autocorr_u[:,1] = np.add(autocorr_u[:,1], autocorr(samp_u,mean_u,std_u))
                autocorr_r[:,1] = np.add(autocorr_r[:,1], autocorr(samp_r,mean_r,std_r))
                spectrum_u[:,1] = np.add(spectrum_u[:,1], np.absolute(np.fft.rfft(samp_u))**2.0)
                spectrum_r[:,1] = np.add(spectrum_r[:,1], np.absolute(np.fft.rfft(samp_r))**2.0)
                done = done + points_per_window
        
        autocorr_u[:,1] = autocorr_u[:,1] / (1.0*N*N_windows)
        autocorr_r[:,1] = autocorr_r[:,1] / (1.0*N*N_windows)
        
        spectrum_u[:,1] = spectrum_u[:,1] / (1.0*N*N_windows)
        spectrum_r[:,1] = spectrum_r[:,1] / (1.0*N*N_windows)
        
        np.savetxt(out_location+"/autocorr_u_"+str(alpha),autocorr_u) 
        np.savetxt(out_location+"/autocorr_r_"+str(alpha),autocorr_r)
        
        np.savetxt(out_location+"/spectrum_u_"+str(alpha),spectrum_u) 
        np.savetxt(out_location+"/spectrum_r_"+str(alpha),spectrum_r)
    

# Call Main
if __name__ == "__main__":
    main()
