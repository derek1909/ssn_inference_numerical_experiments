# This code evolves the network forward in time, collecting samples, 
# to then compute the crosscorrelogram

import numpy as np
import scipy as sp
import methods as mt
from parameters import *
from xcorr_maxlag import xcorr_maxlag

import datetime, os

############
#   Main   #
############

def main():
    
    in_location = "parameter_files"
    
    out_location = "results/correlations_and_power_spectrum/xcorr"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
        
    epsilon = 1.0E-12
    
    #--------------------------#
    #   We import parameters   #
    #--------------------------#
    
    h = np.empty([N_pat,N])
    mu_0 = np.empty([N_pat,N])
    Sigma_0 = np.empty([N_pat,N,N])
    Sigma_eta = np.loadtxt(in_location+"/sigma_eta_learn")
    
    W = np.loadtxt(in_location+"/w_learn")
    
    for alpha in range(N_pat):
        h[alpha] = np.loadtxt(in_location+"/h_true_"+str(alpha)+"_learn")
        mu_0[alpha] = np.loadtxt(in_location+"/mu_evolved_net_"+str(alpha))
        Sigma_0[alpha] = np.loadtxt(in_location+"/sigma_evolved_net_"+str(alpha))
        
       
    #------------------------------#
    #   Evolution of the network   #
    #------------------------------#
   
    total_time = 50000.0 * tau_e # 1000s
    t_bet_samp = 2*dt
    steps_bet_samp = int(t_bet_samp/dt)    
    sample_size = int(total_time/t_bet_samp)
    
    max_time_lag = 5.0*tau_n
    maxlag_bins = int(max_time_lag/t_bet_samp)
    
    taus = np.linspace(-max_time_lag,max_time_lag,num = 2*maxlag_bins + 1, endpoint=True)
    
    E_neurons = np.array([0,5,10,15,20,25,30,35,40,45])
    I_neurons = E_neurons + 50
    
    n_select_neurons = len(E_neurons)
    
    XCorr_EE = np.empty([2*maxlag_bins + 1,n_select_neurons**2+1])
    XCorr_EI = np.empty([2*maxlag_bins + 1,n_select_neurons**2+1])
    XCorr_II = np.empty([2*maxlag_bins + 1,n_select_neurons**2+1])
    
    XCorr_EE_S = np.empty([2*maxlag_bins + 1,n_select_neurons**2+1])
    XCorr_EI_S = np.empty([2*maxlag_bins + 1,n_select_neurons**2+1])
    XCorr_II_S = np.empty([2*maxlag_bins + 1,n_select_neurons**2+1])

    XCorr_EE_AS = np.empty([2*maxlag_bins + 1,n_select_neurons**2+1])
    XCorr_EI_AS = np.empty([2*maxlag_bins + 1,n_select_neurons**2+1])
    XCorr_II_AS = np.empty([2*maxlag_bins + 1,n_select_neurons**2+1])    
    
    XCorr_EE[:,0] = taus
    XCorr_EI[:,0] = taus
    XCorr_II[:,0] = taus
    
    XCorr_EE_S[:,0] = taus
    XCorr_EI_S[:,0] = taus
    XCorr_II_S[:,0] = taus
    
    XCorr_EE_AS[:,0] = taus
    XCorr_EI_AS[:,0] = taus
    XCorr_II_AS[:,0] = taus
    
    print("Start time : ", datetime.datetime.now())
            
    for alpha in range(N_pat):
        u0 = np.random.multivariate_normal(mean = mu_0[alpha], cov = Sigma_0[alpha])
        eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
        
        print("evolving pattern " + str(alpha))
        u, eta = mt.network_evolution(W,h[alpha],u0,Sigma_eta, eta = eta0)
        print("sampling moments")
        (u_samples,_,_,_) = mt.network_sample(W,h[alpha],u,eta,sample_size,steps_bet_samp,Sigma_eta)
        
        #------------------------------#
        #   Xcorrelogram computation   #
        #------------------------------#        
        
        for i in range(n_select_neurons):
            for j in range(n_select_neurons):
                # E-E
                neuron_1 = E_neurons[i]
                neuron_2 = E_neurons[j]
                trace_1 = u_samples[:,neuron_1]
                trace_2 = u_samples[:,neuron_2]
                
                XCorr_EE[:,1+i+n_select_neurons*j] = xcorr_maxlag(trace_1, trace_2, maxlag_bins)
                
                # E-I
                neuron_2 = I_neurons[j]
                trace_2 = u_samples[:,neuron_2]
                XCorr_EI[:,1+i+n_select_neurons*j] = xcorr_maxlag(trace_1, trace_2, maxlag_bins)
                # I-I    
                neuron_1 = I_neurons[i]
                trace_1 = u_samples[:,neuron_1]
                XCorr_II[:,1+i+n_select_neurons*j] = xcorr_maxlag(trace_1, trace_2, maxlag_bins)
        
                                                           
        np.savetxt(out_location+"/XCorrel_net_EE_contrast_"+str(alpha), XCorr_EE)
        np.savetxt(out_location+"/XCorrel_net_EI_contrast_"+str(alpha), XCorr_EI)
        np.savetxt(out_location+"/XCorrel_net_II_contrast_"+str(alpha), XCorr_II)
        
        XCorr_EE_S[:,1:] = 0.5 * (XCorr_EE[:,1:] + np.flip(XCorr_EE[:,1:],0))
        XCorr_EE_AS[:,1:] = 0.5 * (XCorr_EE[:,1:]- np.flip(XCorr_EE[:,1:],0))
        
        XCorr_EI_S[:,1:] = 0.5 * (XCorr_EI[:,1:] + np.flip(XCorr_EI[:,1:],0))
        XCorr_EI_AS[:,1:] = 0.5 * (XCorr_EI[:,1:]- np.flip(XCorr_EI[:,1:],0))
        
        XCorr_II_S[:,1:] = 0.5 * (XCorr_II[:,1:] + np.flip(XCorr_II[:,1:],0))
        XCorr_II_AS[:,1:] = 0.5 * (XCorr_II[:,1:]- np.flip(XCorr_II[:,1:],0))
        
        np.savetxt(out_location+"/S_XCorrel_net_EE_contrast_"+str(alpha), XCorr_EE_S)
        np.savetxt(out_location+"/S_XCorrel_net_EI_contrast_"+str(alpha), XCorr_EI_S)
        np.savetxt(out_location+"/S_XCorrel_net_II_contrast_"+str(alpha), XCorr_II_S)
        np.savetxt(out_location+"/AS_XCorrel_net_EE_contrast_"+str(alpha), XCorr_EE_AS)
        np.savetxt(out_location+"/AS_XCorrel_net_EI_contrast_"+str(alpha), XCorr_EI_AS)
        np.savetxt(out_location+"/AS_XCorrel_net_II_contrast_"+str(alpha), XCorr_II_AS)
        
    print("End time : ", datetime.datetime.now())
        

# Call Main
if __name__ == "__main__":
    main()
