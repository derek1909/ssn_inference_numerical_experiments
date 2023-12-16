# Results for a network of I cells only

import numpy as np
import scipy as sp
import methods as mt
from parameters import *

import datetime, os

def autocorr(x, mean, std):
    x = (x - mean)/std
    result = np.correlate(x, x, "same")/len(x)
    return (result[result.size//2:])

############
#   Main   #
############

def main():

    in_location = "parameter_files"
    
    out_location = "results/I_only_net"
    
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
    
    # Computing average E input into I cells
    nu = np.empty([N_pat,N])
    h_exc = np.empty([N_pat,N_inh])
    for alpha in range(N_pat):
        nu[alpha] = np.loadtxt(in_location+"/nu_evolved_net_"+str(alpha))
        h_exc[alpha] = (W @ nu[alpha])[N_exc:]
    
    
    # Making it an effective-I-only network
    
    h[:,:N_exc] = 0.0
    W[:,:N_exc] = 0.0
    W[:N_exc,N_exc:] = 0.0
    
    for alpha in range(N_pat):
        h[alpha,N_exc:] += h_exc[alpha]
    
    np.savetxt(out_location + "/W", W)
    for alpha in range(N_pat):
        np.savetxt(out_location + "/h_"+str(alpha),h[alpha])
        
    #------------------------------#
    #   Evolution of the network   #
    #------------------------------#
           
    mu_0 = np.empty([N_pat,N])
    Sigma_0 = np.empty([N_pat,N,N])
           
    for alpha in range(N_pat):
        mu_0[alpha] = np.loadtxt(in_location+"/mu_learn_"+str(alpha))
        Sigma_0[alpha] = np.loadtxt(in_location+"/sigma_learn_"+str(alpha))
        
    mu_final = np.empty([N_pat,N])
    Sigma_final = np.empty([N_pat,N,N])
    
    nu_final = np.empty([N_pat,N])
    Lambda_final = np.empty([N_pat,N,N])
    
    total_time = 50000.0 * tau_e # 1000s
    t_bet_samp = 2 * dt
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
    
    print("Start time : ", datetime.datetime.now())
    
        
    for alpha in range(N_pat):
        u0 = np.random.multivariate_normal(mean = mu_0[alpha], cov = Sigma_0[alpha])
        eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
        
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
    
        autocorr_u[:,1] = np.zeros([times])
        autocorr_r[:,1] = np.zeros([times])
        autocorr_LFP[:,1] = np.zeros([times])
        
        spectrum_u[:,1] = np.zeros([freqs])
        spectrum_r[:,1] = np.zeros([freqs])
        spectrum_LFP[:,1] = np.zeros([freqs])
        
        autocorr_u_indiv[:,:,1] = np.zeros([N,times])
        autocorr_r_indiv[:,:,1] = np.zeros([N,times])
        
        LFP = np.sum(u_samples[:,N_exc:], axis=1)
        
        done = 0
        mean_LFP = np.mean(LFP)
        std_LFP = np.std(LFP)
        for win in range(N_windows):
            samp_LFP = LFP[done:done+points_per_window]
            autocorr_LFP[:,1] += autocorr(samp_LFP,mean_LFP,std_LFP)
            spectrum_LFP[:,1] = np.add(spectrum_LFP[:,1], 
                                    np.absolute(np.fft.rfft(samp_LFP-mean_LFP))**2.0)
            done = done + points_per_window
        
        for i in range(N_exc,N):
            done = 0
            mean_u = np.mean(u_samples[:,i])
            std_u = np.std(u_samples[:,i])
            mean_r = np.mean(r_samples[:,i])
            std_r = np.std(r_samples[:,i])
            
                                    
            for win in range(N_windows):
                samp_u = u_samples[done:done+points_per_window,i]
                samp_r = r_samples[done:done+points_per_window,i]
                autocorr_u_indiv[i,:,1] += autocorr(samp_u,mean_u,std_u)
                autocorr_r_indiv[i,:,1] += autocorr(samp_r,mean_r,std_r)
                spectrum_u[:,1] = np.add(spectrum_u[:,1], np.absolute(np.fft.rfft(samp_u-mean_u))**2.0)
                spectrum_r[:,1] = np.add(spectrum_r[:,1], np.absolute(np.fft.rfft(samp_r-mean_r))**2.0)
                done = done + points_per_window
            autocorr_u[:,1] += autocorr_u_indiv[i,:,1]
            autocorr_r[:,1] += autocorr_r_indiv[i,:,1]
        
                
        autocorr_u_indiv[:,:,1] = autocorr_u_indiv[:,:,1] / (1.0*N_windows)
        autocorr_r_indiv[:,:,1] = autocorr_r_indiv[:,:,1] / (1.0*N_windows)
            
        autocorr_u[:,1] = autocorr_u[:,1] / (1.0*N_exc*N_windows)
        autocorr_r[:,1] = autocorr_r[:,1] / (1.0*N_exc*N_windows)
        autocorr_LFP[:,1] = autocorr_LFP[:,1] / (1.0*N_windows)
        
        spectrum_u[:,1] = spectrum_u[:,1] / (1.0*N_exc*N_windows)
        spectrum_r[:,1] = spectrum_r[:,1] / (1.0*N_exc*N_windows)
        spectrum_LFP[:,1] = spectrum_LFP[:,1] / (1.0*N_windows)
        
        np.savetxt(out_location+"/autocorr_u_"+str(alpha),autocorr_u) 
        np.savetxt(out_location+"/autocorr_r_"+str(alpha),autocorr_r)
        np.savetxt(out_location+"/autocorr_LFP_"+str(alpha),autocorr_LFP)
        
        for i in range(N_exc,N):
            np.savetxt(out_location+"/autocorr_u_"+str(alpha)+"_cell_"+str(i) ,autocorr_u_indiv[i]) 
            np.savetxt(out_location+"/autocorr_r_"+str(alpha)+"_cell_"+str(i) ,autocorr_r_indiv[i])
        
        np.savetxt(out_location+"/spectrum_u_"+str(alpha),spectrum_u) 
        np.savetxt(out_location+"/spectrum_r_"+str(alpha),spectrum_r)
        np.savetxt(out_location+"/spectrum_LFP_"+str(alpha),spectrum_LFP)
    
                
    print("End time : ", datetime.datetime.now())
        

# Call Main
if __name__ == "__main__":
    main()
