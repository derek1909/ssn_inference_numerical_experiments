import numpy as np
import scipy as sp
import scipy.interpolate, scipy.signal
import methods as mt
from parameters import *

import datetime, os

def SKL_2_Gaussians(mu1,mu2,Sigma1,Sigma2):
    Sigma1_inv = np.linalg.inv(Sigma1)
    Sigma2_inv = np.linalg.inv(Sigma2)
    Sigma_inv = Sigma1_inv + Sigma2_inv
    err_mean = 0.25* np.dot(mu1-mu2,np.dot(Sigma_inv, mu1-mu2))
    if (err_mean < 0):
        print("ERROR MEAN IS NEG!!!")
    err_cov = 0.25* ( np.trace(Sigma2_inv @ Sigma1) +
                    np.trace(Sigma1_inv @ Sigma2)) - 0.5* len(mu1)
    if (err_cov < 0):
        print("ERROR COV IS NEG!!!")
    
    tot_err = err_mean + err_cov
    
    return (tot_err, err_mean, err_cov)



############
#   Main   #
############

def main():
    
    in_location_net = "parameter_files"
    in_location_train = "results/div_norm_data/z_from_gamma/train"
    in_location_test = "results/div_norm_data/z_from_gamma/test"
    
    out_location = "results/SKLs"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)  
        
    n_tot_pat = 1000
       
    all_patterns = np.arange(n_tot_pat)
    z_array = np.empty([n_tot_pat])
    
    for i in range(n_tot_pat):
        if (i<500):
            z_array[i] = np.loadtxt(in_location_train+"/z_"+str(i))
        else:
            z_array[i] = np.loadtxt(in_location_test+"/z_"+str(i-500))
    
    used_patterns = all_patterns[z_array<=1.0]
    n_pat = len(used_patterns)
    print("Using ",n_pat," patterns")
    
    mu_SSN_array = np.empty([n_pat,N_exc])
    mu_GSM_array = np.empty([n_pat,N_exc])
    Sigma_SSN_array = np.empty([n_pat,N_exc,N_exc])
    Sigma_GSM_array = np.empty([n_pat,N_exc,N_exc])
    
    for alpha in range(n_pat):
        if (used_patterns[alpha]<500):
            mu_SSN_array[alpha] = np.loadtxt(in_location_train+"/mu_SSN_"+str(used_patterns[alpha]))[:N_exc]
            mu_GSM_array[alpha] = np.loadtxt(in_location_train+"/mu_GSM_"+str(used_patterns[alpha]))[:N_exc]
            
            Sigma_SSN_array[alpha] = np.loadtxt(in_location_train+"/Sigma_SSN_"+str(used_patterns[alpha]))[:N_exc,:N_exc]
            Sigma_GSM_array[alpha] = np.loadtxt(in_location_train+"/Sigma_GSM_"+str(used_patterns[alpha]))[:N_exc,:N_exc]
        else:    
            mu_SSN_array[alpha] = np.loadtxt(in_location_test+"/mu_SSN_"+str(used_patterns[alpha]-500))[:N_exc]
            mu_GSM_array[alpha] = np.loadtxt(in_location_test+"/mu_GSM_"+str(used_patterns[alpha]-500))[:N_exc]
            
            Sigma_SSN_array[alpha] = np.loadtxt(in_location_test+"/Sigma_SSN_"+str(used_patterns[alpha]-500))[:N_exc,:N_exc]
            Sigma_GSM_array[alpha] = np.loadtxt(in_location_test+"/Sigma_GSM_"+str(used_patterns[alpha]-500))[:N_exc,:N_exc]
    
    done_SSN_vs_GSM_other = 0
    done_SSN_vs_SSN_other = 0
    
    n_pairs_SSN_vs_GSM = n_pat * (n_pat-1)
    n_pairs_SSN_vs_SSN = (n_pat * (n_pat-1))//2
    
    SKL_array_SSN_vs_GSM_match = np.empty([n_pat,4])
    SKL_array_SSN_vs_GSM_other = np.empty([n_pairs_SSN_vs_GSM,5])
    SKL_array_SSN_vs_SSN_other = np.empty([n_pairs_SSN_vs_SSN,5])
    SKL_array_GSM_vs_GSM_other = np.empty([n_pairs_SSN_vs_SSN,5])
    
    for alpha in range(n_pat):
        SKL_array_SSN_vs_GSM_match[alpha,0] = alpha
        SKL_array_SSN_vs_GSM_match[alpha,1:] = SKL_2_Gaussians(
                                                        mu_SSN_array[alpha],mu_GSM_array[alpha],
                                                        Sigma_SSN_array[alpha],Sigma_GSM_array[alpha])
        
        for beta in range(n_pat):
            if not (beta == alpha):
                SKL_array_SSN_vs_GSM_other[done_SSN_vs_GSM_other,0] = alpha
                SKL_array_SSN_vs_GSM_other[done_SSN_vs_GSM_other,1] = beta         
                SKL_array_SSN_vs_GSM_other[done_SSN_vs_GSM_other,2:] = SKL_2_Gaussians(
                                                        mu_SSN_array[alpha],mu_GSM_array[beta],
                                                        Sigma_SSN_array[alpha],Sigma_GSM_array[beta])
                done_SSN_vs_GSM_other += 1
                
            if (beta < alpha):    
                SKL_array_SSN_vs_SSN_other[done_SSN_vs_SSN_other,0] = alpha
                SKL_array_SSN_vs_SSN_other[done_SSN_vs_SSN_other,1] = beta         
                SKL_array_SSN_vs_SSN_other[done_SSN_vs_SSN_other,2:] = SKL_2_Gaussians(
                                                        mu_SSN_array[alpha],mu_SSN_array[beta],
                                                        Sigma_SSN_array[alpha],Sigma_SSN_array[beta])
                
                SKL_array_GSM_vs_GSM_other[done_SSN_vs_SSN_other,0] = alpha
                SKL_array_GSM_vs_GSM_other[done_SSN_vs_SSN_other,1] = beta         
                SKL_array_GSM_vs_GSM_other[done_SSN_vs_SSN_other,2:] = SKL_2_Gaussians(
                                                        mu_GSM_array[alpha],mu_GSM_array[beta],
                                                        Sigma_GSM_array[alpha],Sigma_GSM_array[beta])
                
                done_SSN_vs_SSN_other += 1
            
    
    n_bins_histo = 50#int(np.sqrt(n_pat))
    
            
    max_val_tot = 160.0 #max(np.amax(SKL_array_SSN_vs_GSM_match[:,1]),np.amax(SKL_array_SSN_vs_GSM_other[:,2]),np.amax(SKL_array_SSN_vs_SSN_other[:,2]))
    max_val_cov = max_val_tot
    max_val_mean = 5.0
    
    histo_SKL_total_SSN_vs_GSM_match = np.empty([n_bins_histo,2])
    histo_SKL_mean_SSN_vs_GSM_match = np.empty([n_bins_histo,2])
    histo_SKL_cov_SSN_vs_GSM_match = np.empty([n_bins_histo,2])        
    
    histo_SKL_total_SSN_vs_GSM_other = np.empty([n_bins_histo,2])
    histo_SKL_mean_SSN_vs_GSM_other = np.empty([n_bins_histo,2])
    histo_SKL_cov_SSN_vs_GSM_other = np.empty([n_bins_histo,2]) 
    
    histo_SKL_total_SSN_vs_SSN_other = np.empty([n_bins_histo,2])
    histo_SKL_mean_SSN_vs_SSN_other = np.empty([n_bins_histo,2])
    histo_SKL_cov_SSN_vs_SSN_other = np.empty([n_bins_histo,2])
    
    histo_SKL_total_GSM_vs_GSM_other = np.empty([n_bins_histo,2])
    histo_SKL_mean_GSM_vs_GSM_other = np.empty([n_bins_histo,2])
    histo_SKL_cov_GSM_vs_GSM_other = np.empty([n_bins_histo,2])
    
    histo_SKL_total_SSN_vs_GSM_match[:,1],bin_edges_tot = np.histogram(SKL_array_SSN_vs_GSM_match[:,1], bins=n_bins_histo, range=(0,max_val_tot), density=True)
    
    histo_SKL_mean_SSN_vs_GSM_match[:,1],bin_edges_mean = np.histogram(SKL_array_SSN_vs_GSM_match[:,2], bins=n_bins_histo, range=(0,max_val_mean), density=True)
    
    histo_SKL_cov_SSN_vs_GSM_match[:,1],bin_edges_cov = np.histogram(SKL_array_SSN_vs_GSM_match[:,3], bins=n_bins_histo, range=(0,max_val_cov), density=True)
    
    
    histo_SKL_total_SSN_vs_GSM_other[:,1],bin_edges = np.histogram(SKL_array_SSN_vs_GSM_other[:,2], bins=n_bins_histo, range=(0,max_val_tot), density=True)
    
    histo_SKL_mean_SSN_vs_GSM_other[:,1],_ = np.histogram(SKL_array_SSN_vs_GSM_other[:,3], bins=n_bins_histo, range=(0,max_val_mean), density=True)
    
    histo_SKL_cov_SSN_vs_GSM_other[:,1],_ = np.histogram(SKL_array_SSN_vs_GSM_other[:,4], bins=n_bins_histo, range=(0,max_val_cov), density=True)
    
    
    histo_SKL_total_SSN_vs_SSN_other[:,1],bin_edges = np.histogram(SKL_array_SSN_vs_SSN_other[:,2], bins=n_bins_histo, range=(0,max_val_tot), density=True)
    
    histo_SKL_mean_SSN_vs_SSN_other[:,1],_ = np.histogram(SKL_array_SSN_vs_SSN_other[:,3], bins=n_bins_histo, range=(0,max_val_mean), density=True)
    
    histo_SKL_cov_SSN_vs_SSN_other[:,1],_ = np.histogram(SKL_array_SSN_vs_SSN_other[:,4], bins=n_bins_histo, range=(0,max_val_cov), density=True)
    
    
    histo_SKL_total_GSM_vs_GSM_other[:,1],bin_edges = np.histogram(SKL_array_GSM_vs_GSM_other[:,2], bins=n_bins_histo, range=(0,max_val_tot), density=True)
    
    histo_SKL_mean_GSM_vs_GSM_other[:,1],_ = np.histogram(SKL_array_GSM_vs_GSM_other[:,3], bins=n_bins_histo, range=(0,max_val_mean), density=True)
    
    histo_SKL_cov_GSM_vs_GSM_other[:,1],_ = np.histogram(SKL_array_GSM_vs_GSM_other[:,4], bins=n_bins_histo, range=(0,max_val_cov), density=True)
    
    
    bin_centers_tot = 0.5*(bin_edges_tot[:-1]+bin_edges_tot[1:])
    bin_centers_mean = 0.5*(bin_edges_mean[:-1]+bin_edges_mean[1:])
    bin_centers_cov = 0.5*(bin_edges_cov[:-1]+bin_edges_cov[1:])
    
    histo_SKL_total_SSN_vs_GSM_match[:,0] = bin_centers_tot
    histo_SKL_mean_SSN_vs_GSM_match[:,0] = bin_centers_mean
    histo_SKL_cov_SSN_vs_GSM_match[:,0] = bin_centers_cov
    
    histo_SKL_total_SSN_vs_GSM_other[:,0] = bin_centers_tot
    histo_SKL_mean_SSN_vs_GSM_other[:,0] = bin_centers_mean
    histo_SKL_cov_SSN_vs_GSM_other[:,0] = bin_centers_cov
    
    histo_SKL_total_SSN_vs_SSN_other[:,0] = bin_centers_tot
    histo_SKL_mean_SSN_vs_SSN_other[:,0] = bin_centers_mean
    histo_SKL_cov_SSN_vs_SSN_other[:,0] = bin_centers_cov
    
    histo_SKL_total_GSM_vs_GSM_other[:,0] = bin_centers_tot
    histo_SKL_mean_GSM_vs_GSM_other[:,0] = bin_centers_mean
    histo_SKL_cov_GSM_vs_GSM_other[:,0] = bin_centers_cov
    
    np.save(out_location+"/SKL_array_SSN_vs_GSM_match.npy", SKL_array_SSN_vs_GSM_match)
    np.save(out_location+"/SKL_array_SSN_vs_GSM_other.npy", SKL_array_SSN_vs_GSM_other)
    np.save(out_location+"/SKL_array_SSN_vs_SSN_other.npy", SKL_array_SSN_vs_SSN_other)
    np.save(out_location+"/SKL_array_GSM_vs_GSM_other.npy", SKL_array_GSM_vs_GSM_other)
    
    np.savetxt(out_location+"/histo_SKL_total_SSN_vs_GSM_match", histo_SKL_total_SSN_vs_GSM_match)
    np.savetxt(out_location+"/histo_SKL_mean_SSN_vs_GSM_match", histo_SKL_mean_SSN_vs_GSM_match)
    np.savetxt(out_location+"/histo_SKL_cov_SSN_vs_GSM_match", histo_SKL_cov_SSN_vs_GSM_match)
    
    np.savetxt(out_location+"/histo_SKL_total_SSN_vs_GSM_other", histo_SKL_total_SSN_vs_GSM_other)
    np.savetxt(out_location+"/histo_SKL_mean_SSN_vs_GSM_other", histo_SKL_mean_SSN_vs_GSM_other)
    np.savetxt(out_location+"/histo_SKL_cov_SSN_vs_GSM_other", histo_SKL_cov_SSN_vs_GSM_other)
    
    np.savetxt(out_location+"/histo_SKL_total_SSN_vs_SSN_other", histo_SKL_total_SSN_vs_SSN_other)
    np.savetxt(out_location+"/histo_SKL_mean_SSN_vs_SSN_other", histo_SKL_mean_SSN_vs_SSN_other)
    np.savetxt(out_location+"/histo_SKL_cov_SSN_vs_SSN_other", histo_SKL_cov_SSN_vs_SSN_other)
    
    np.savetxt(out_location+"/histo_SKL_total_GSM_vs_GSM_other", histo_SKL_total_GSM_vs_GSM_other)
    np.savetxt(out_location+"/histo_SKL_mean_GSM_vs_GSM_other", histo_SKL_mean_GSM_vs_GSM_other)
    np.savetxt(out_location+"/histo_SKL_cov_GSM_vs_GSM_other", histo_SKL_cov_GSM_vs_GSM_other)
    
# Call Main
if __name__ == "__main__":
    main()
