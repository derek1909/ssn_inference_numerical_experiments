# This code runs a Langevin sampler to then compute the crosscorrelogram

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
    
    in_location_net = "parameter_files"
    in_location_langevin = "results/langevin"
    out_location = in_location_langevin + "/xcorr"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
        
    epsilon = 1.0E-12
    
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
    #   Evolution of the network   #
    #------------------------------#
           
    total_time = 500000.0 * tau_e # 10000s
    t_bet_samp = 20*dt
    steps_bet_samp = int(t_bet_samp/dt)    
    sample_size = int(total_time/t_bet_samp)
    
    max_time_lag = 50.0*tau_n
    maxlag_bins = int(max_time_lag/t_bet_samp)
    
    taus = np.linspace(-max_time_lag,max_time_lag,num = 2*maxlag_bins + 1, endpoint=True)
    
    E_neurons = np.array([0,5,10,15,20,25,30,35,40,45])
    I_neurons = E_neurons+50
    
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
        print("sampling pattern " + str(alpha))
        mu_ext = np.concatenate((mu_net[alpha],np.zeros(N)))
        Sigma_solve = np.loadtxt(in_location_langevin+"/Sigma_solve_"+str(alpha))
        init = np.random.multivariate_normal(mean = mu_ext, cov = Sigma_solve)
        u0 = init[:N]
        eta0 = init[N:]
        
        (_,_,u,eta) = mt.Langevin_sample(A[alpha],B,mu_net[alpha],
                                           1000,steps_bet_samp,
                                           u0,eta0)
        (u_samples,_,_,_) = mt.Langevin_sample(A[alpha],B,mu_net[alpha],
                                           sample_size,steps_bet_samp,
                                           u,eta)
        
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
        
                                                           
        np.savetxt(out_location+"/XCorrel_Langevin_EE_contrast_"+str(alpha), XCorr_EE)
        np.savetxt(out_location+"/XCorrel_Langevin_EI_contrast_"+str(alpha), XCorr_EI)
        np.savetxt(out_location+"/XCorrel_Langevin_II_contrast_"+str(alpha), XCorr_II)

        XCorr_EE_S[:,1:] = 0.5 * (XCorr_EE[:,1:] + np.flip(XCorr_EE[:,1:],0))
        XCorr_EE_AS[:,1:] = 0.5 * (XCorr_EE[:,1:]- np.flip(XCorr_EE[:,1:],0))
        
        XCorr_EI_S[:,1:] = 0.5 * (XCorr_EI[:,1:] + np.flip(XCorr_EI[:,1:],0))
        XCorr_EI_AS[:,1:] = 0.5 * (XCorr_EI[:,1:]- np.flip(XCorr_EI[:,1:],0))
        
        XCorr_II_S[:,1:] = 0.5 * (XCorr_II[:,1:] + np.flip(XCorr_II[:,1:],0))
        XCorr_II_AS[:,1:] = 0.5 * (XCorr_II[:,1:]- np.flip(XCorr_II[:,1:],0))
        
        np.savetxt(out_location+"/S_XCorrel_Langevin_EE_contrast_"+str(alpha), XCorr_EE_S)
        np.savetxt(out_location+"/S_XCorrel_Langevin_EI_contrast_"+str(alpha), XCorr_EI_S)
        np.savetxt(out_location+"/S_XCorrel_Langevin_II_contrast_"+str(alpha), XCorr_II_S)
        np.savetxt(out_location+"/AS_XCorrel_Langevin_EE_contrast_"+str(alpha), XCorr_EE_AS)
        np.savetxt(out_location+"/AS_XCorrel_Langevin_EI_contrast_"+str(alpha), XCorr_EI_AS)
        np.savetxt(out_location+"/AS_XCorrel_Langevin_II_contrast_"+str(alpha), XCorr_II_AS)

    
    print("End time : ", datetime.datetime.now())
        

# Call Main
if __name__ == "__main__":
    main()
