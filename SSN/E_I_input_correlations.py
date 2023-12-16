# This code computes the auto and xcorrelations of E and I inputs to each cell in the network

import numpy as np
import scipy as sp
import methods as mt
from parameters import *
import datetime, os, sys

def xcorr_maxlag(x, y, maxlag=1.0):
    xl = x.size
    yl = y.size

    c = np.zeros(2*maxlag + 1)

    for i in range(maxlag+1):
        tmp = np.corrcoef(x[0:min(xl, yl-i)], y[i:i+min(xl, yl-i)])
        c[maxlag-i] = tmp[1][0]
        tmp = np.corrcoef(x[i:i+min(xl-i, yl)], y[0:min(xl-i, yl)])
        c[maxlag+i] = tmp[1][0]

    return c
    
############
#   Main   #
############

def main():
    lagged_noise = False
    
    net_params_location = "parameter_files"
    
    if (lagged_noise == True):
        print("########################")
        print("#  USING LAGGED NOISE  #")
        print("########################")
        tag = "_w_lagged_noise"
        lag_time = tau_e/5.0 # 4ms
        buffer_size =int(lag_time/dt)
        in_location = "results/correlations_and_power_spectrum_HD"
        out_location = "results/EI_input_xcorrel_HD"
    else:
        in_location = "results/correlations_and_power_spectrum_UHD"
        out_location = "results/EI_input_xcorrel_UHD"
    
    
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    
    total_time = 10000.0 * tau_e # 400s
    t_bet_samp = dt
    steps_bet_samp = 1  
    sample_size = int(total_time/t_bet_samp)
    
    
    max_time_lag = 0.5*tau_n
    maxlag_bins = int(max_time_lag/t_bet_samp)
    taus = np.empty([2*maxlag_bins + 1,1])
    
    for t in range(taus.size):
        taus[t,0] =  ((1.0*t)/(1.0*maxlag_bins))-1 
    
    taus = taus * max_time_lag
    
    XCorr_rec = np.empty([N,2*maxlag_bins + 1, 3])
    XCorr_all = np.empty([N,2*maxlag_bins + 1, 3])
    
    lag_array_rec = np.empty([N,N_pat])
    max_xcorr_array_rec = np.empty([N,N_pat])
    
    lag_array_all = np.empty([N,N_pat])
    max_xcorr_array_all = np.empty([N,N_pat])
    
    neuron_list = np.array([0,25,50,75])
    
    W = np.loadtxt(net_params_location+"/w_learn")
    
    print("Start time : ", datetime.datetime.now())
    print("Working with ", N_pat, " input levels")   
    
    
    for alpha in range(N_pat):
        print("Input level " + str(alpha))
        h = np.expand_dims(np.loadtxt(net_params_location+"/h_true_"+str(alpha)+"_learn"),0)
        print("Loading samples...")
        r_samples = np.load(in_location+"/r_samples_"+str(alpha)+".npy")
        eta_samples = np.load(in_location+"/eta_samples_"+str(alpha)+".npy")
        
        r_samples_e = r_samples[:,:N_exc]
        r_samples_i = r_samples[:,N_exc:]
        
        external_input = h + eta_samples
        
        positive_input_indexes = external_input >= 0.0
        negative_input_indexes = external_input < 0.0
        
        
        W_e_T = (W[:,:N_exc]).T
        W_i_T = (W[:,N_exc:]).T
        print("Computing conductance history...")
        
        # Recurrent
        g_e_rec = r_samples_e @ W_e_T # Size = samples x N
        g_i_rec = r_samples_i @ W_i_T # Size = samples x N
        
        # External
        g_e_ext = np.zeros([sample_size,N])
        g_i_ext = np.zeros([sample_size,N])
        
        g_e_ext[positive_input_indexes] = external_input[positive_input_indexes]
        g_i_ext[negative_input_indexes] = external_input[negative_input_indexes]
        
        # All
        g_e = g_e_rec + g_e_ext
        g_i = g_i_rec + g_i_ext
        
        print("Cumputing correlations...")
        for i in range(N):
            e_cond_rec = g_e_rec[:,i]
            i_cond_rec = g_i_rec[:,i]
            e_cond_all = g_e[:,i]
            i_cond_all = g_i[:,i]
            XCorr_rec[i,:,0] = xcorr_maxlag(e_cond_rec, e_cond_rec, maxlag_bins)    
            XCorr_rec[i,:,1] = xcorr_maxlag(i_cond_rec, i_cond_rec, maxlag_bins)
            XCorr_rec[i,:,2] = xcorr_maxlag(e_cond_rec, i_cond_rec, maxlag_bins)
            
            lag_array_rec[i,alpha] = taus[np.argmin(XCorr_rec[i,:,2]),0]
            max_xcorr_array_rec[i,alpha] = np.amin(XCorr_rec[i,:,2])
             
            XCorr_all[i,:,0] = xcorr_maxlag(e_cond_all, e_cond_all, maxlag_bins)    
            XCorr_all[i,:,1] = xcorr_maxlag(i_cond_all, i_cond_all, maxlag_bins)
            XCorr_all[i,:,2] = xcorr_maxlag(e_cond_all, i_cond_all, maxlag_bins) 
            
            lag_array_all[i,alpha] = taus[np.argmin(XCorr_all[i,:,2]),0]
            max_xcorr_array_all[i,alpha] = np.amin(XCorr_all[i,:,2])
            
        np.save(out_location+"/XCorr_all_"+str(alpha), XCorr_all)    
        np.save(out_location+"/XCorr_rec_"+str(alpha), XCorr_rec)
        
        
        for neuron in neuron_list:
            x_correl_w_taus_all = np.concatenate((taus,XCorr_all[neuron]),1)
            np.savetxt(out_location+"/XCorrel_EI_to_neuron_all_"+str(neuron)+"_contrast_"+str(alpha), x_correl_w_taus_all)
            x_correl_w_taus_rec = np.concatenate((taus,XCorr_rec[neuron]),1)
            np.savetxt(out_location+"/XCorrel_EI_to_neuron_rec_"+str(neuron)+"_contrast_"+str(alpha), x_correl_w_taus_rec)          
    np.savetxt(out_location+"/lag_vs_contrast_rec", lag_array_rec)
    np.savetxt(out_location+"/max_xcorr_vs_contrast_rec", max_xcorr_array_rec)
    
    np.savetxt(out_location+"/lag_vs_contrast_all", lag_array_all)
    np.savetxt(out_location+"/max_xcorr_vs_contrast_all", max_xcorr_array_all)
    
    print("Final time : ", datetime.datetime.now())                  

# Call Main
if __name__ == "__main__":
    main()
