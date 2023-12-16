# This code computes the power spectra along PCs of the covariance matrix

import numpy as np
import scipy as sp
import methods as mt
from parameters import *
import datetime, os, sys


# Definition of autocorrelation
    
def get_autocorr(x, mean, std):
    x = (x - mean)/std
    result = np.correlate(x, x, "same")/len(x)
    return (result[result.size//2:])

############
#   Main   #
############

def main():
    
    in_location_net = "results"
    in_location_patterns = in_location_net + "/div_norm_data/z_from_gamma/test"
    
    out_location = in_location_net+"/power_spectrum_along_PCS"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)  
    
    # We look for the first 50 patterns with contrasts in the trainning range    
    
    n_pat = 50
    n_candidates = 2*n_pat # it should be > 2 * n_pat
    z = np.empty([n_candidates])  
    candidates = np.arange(n_candidates)
    for alpha in range(n_candidates):
        z[alpha] = np.loadtxt(in_location_patterns+"/z_"+str(alpha))
    
    reselect_patterns = z <= 1.0
    
    patterns = candidates[reselect_patterns]
    patterns = patterns[:n_pat]
    n_pat_check = len(patterns)
    
    if (not n_pat_check == n_pat):
        print("Error! Wrong number of patterns. N_pat_check = ", n_pat_check)
        sys.exit()
    else:
        np.savetxt(out_location+"/used_patterns",patterns)
        
    select_PCs = np.arange(N_exc)
    
    
    total_time = 50000.0 * tau_e # 1000s
    t_bet_samp = 2*dt
    steps_bet_samp = int(t_bet_samp/dt)    
    sample_size = int(total_time/t_bet_samp)
    
    T_window = 100.0 * tau_e # 2s
    points_per_window = int(T_window/t_bet_samp)
    
    N_windows = int(total_time/T_window) 
    
    ts = t_bet_samp
    fs = 1.0/T_window
    
    times = points_per_window//2 
    freqs = points_per_window//2 + 1
    
    r_samples = np.empty([n_pat,sample_size,N])
    
    z = np.empty([n_pat])
    h = np.empty([n_pat,N])
    mu_0 = np.empty([n_pat,N])
    Sigma_0 = np.empty([n_pat,N,N])
    
    eigenvalues = np.empty([n_pat,N_exc])
    eigenvectors = np.empty([n_pat,N_exc,N_exc])
    check = np.empty([N,3])
    spectrum = np.zeros([freqs,2])
    autocorr = np.zeros([times,2])
    for i in range(freqs):    
        spectrum[i,0] = fs * i
    for i in range(times):
        autocorr[i,0] = ts * i
        
    Sigma_eta = np.loadtxt(in_location_net+"/sigma_eta_learn")
    W = np.loadtxt(in_location_net+"/w_learn")
    
    print("Start time : ", datetime.datetime.now())
    
    for alpha in range(n_pat):
        z[alpha] = np.loadtxt(in_location_patterns+"/z_"+str(patterns[alpha]))
        print("Contrast of pattern ",patterns[alpha]," = ", z[alpha])
        h[alpha] = np.loadtxt(in_location_patterns +"/h"+str(patterns[alpha]))
        mu_0[alpha] = np.loadtxt(in_location_patterns+"/mu_SSN_"+str(patterns[alpha]))
        Sigma_0[alpha] = np.loadtxt(in_location_patterns+"/Sigma_SSN_"+str(patterns[alpha]))
        
        np.savetxt(out_location+"/z_"+str(patterns[alpha]),np.expand_dims(z[alpha], axis = 0))
        np.savetxt(out_location+"/h"+str(patterns[alpha]),h[alpha])
        np.savetxt(out_location+"/mu_SSN_"+str(patterns[alpha]),mu_0[alpha])
        np.savetxt(out_location+"/Sigma_SSN_"+str(patterns[alpha]),Sigma_0[alpha])
        
        # Doing PCA on Sigma
        
        Sigma_ = Sigma_0[alpha,:N_exc,:N_exc] #- np.mean(Sigma_0[alpha])
        
        eivals, eivecs = np.linalg.eigh(Sigma_)
                
        idx = eivals.argsort()[::-1] 
        eigenvalues[alpha] = np.copy(eivals[idx])
        eigenvectors[alpha] = np.copy(eivecs[:,idx])
        
        np.savetxt(out_location+"/eigenvalues_Sigma_"+str(patterns[alpha]),eigenvalues[alpha])
        np.savetxt(out_location+"/eigenvectors_Sigma_"+str(patterns[alpha]),eigenvectors[alpha])         
     
    for alpha in range(n_pat):   
        u0 = np.random.multivariate_normal(mean = mu_0[alpha], cov = Sigma_0[alpha])
        eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
        
        print("evolving pattern" + str(alpha))
        u, eta = mt.network_evolution(W,h[alpha],u0,Sigma_eta, eta = eta0)
        print("sampling from the network")
        (u_samples,eta_samples,_,_) = mt.network_sample(W,h[alpha],u,eta,sample_size,
                                                steps_bet_samp,Sigma_eta)
        
        #r_samples = mt.get_r(u_samples)
            
        #np.save(out_location+"/u_samples_"+str(patterns[alpha]),u_samples)
        #np.save(out_location+"/r_samples_"+str(patterns[alpha]),r_samples)
        #np.save(out_location+"/eta_samples_"+str(patterns[alpha]), eta_samples)
        
        for beta in range(n_pat):
            for pc in select_PCs:
                u_in_PC = np.dot(u_samples[:,:N_exc],eigenvectors[beta,:,pc])
                u_in_PC -= np.mean(u_in_PC)
                spectrum[:,1] = np.zeros([freqs])
                autocorr[:,1] = np.zeros([times])
                done = 0
                mean_u_in_PC = np.mean(u_in_PC)
                std_u_in_PC = np.std(u_in_PC)
                for win in range(N_windows):
                    samp = u_in_PC[done:done+points_per_window]
                    done = done + points_per_window
                    spectrum[:,1] = np.add(spectrum[:,1], np.absolute(np.fft.rfft(samp))**2.0)
                    autocorr[:,1] += get_autocorr(samp,mean_u_in_PC,std_u_in_PC)
                
                spectrum[:,1] /= (1.0*N_windows)
                autocorr[:,1] /= (1.0*N_windows)
                
                np.savetxt(out_location+"/spectrum_pat_"+str(patterns[alpha])+"_in_PC_"+str(pc)+"_of_pat_"+str(patterns[beta]),spectrum) 
                np.savetxt(out_location+"/autocorr_pat_"+str(patterns[alpha])+"_in_PC_"+str(pc)+"_of_pat_"+str(patterns[beta]),autocorr)
                
                if (beta == alpha):
                    check[pc,0] = pc
                    check[pc,1] = eigenvalues[beta,pc]
                    check[pc,2] = 0.5*np.sum(spectrum[1:,1])/(freqs**2)
                
            np.savetxt(out_location+"/check_spectrum_pat_"+str(patterns[alpha]),check)                                                     
    print("End time : ", datetime.datetime.now())
        

# Call Main
if __name__ == "__main__":
    main()
