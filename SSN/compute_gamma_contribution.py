import numpy as np
import scipy as sp
import scipy.interpolate, scipy.signal
import methods as mt
from parameters import *

import datetime, os

def smoothen(x,y,xx):
    f = scipy.interpolate.interp1d(x,y)
    yy = f(xx)
    window = scipy.signal.gaussian(100, 20)
    smoothed = scipy.signal.convolve(yy, window/window.sum(), mode='same')
    return smoothed


############
#   Main   #
############

def main():
    
    in_location = "results/power_spectrum_along_PCS"
        
    out_location = in_location + "/no_gamma"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)  
        
    n_points = 2000
    
    gamma_range = np.array([20,80])
    
    range_1 = np.array([1,10])
    range_2 = np.array([175,190])
    
    cut_freq = 1000 # it should be a power of 10 to avoid rounding errors
    
    patterns = np.loadtxt(in_location+"/used_patterns").astype(int)
    print("Loading patterns: ", patterns)
    n_pat = len(patterns)
    
    n_pcs = 10
    select_PCs = np.arange(n_pcs)
    
    eigenvalues = np.empty([n_pat,N_exc])
    eigenvectors = np.empty([n_pat,N_exc,N_exc])
   
    gamma_contribution = np.empty([n_pcs,9])
    
    gamma_contribution_self_all = np.empty([n_pcs*n_pat,9])
    gamma_contribution_other_all = np.empty([n_pcs*n_pat*(n_pat-1),9])
    
    mean_gamma_contribution_self = np.zeros((n_pcs,5))
    mean_gamma_contribution_other = np.zeros((n_pcs,5))
    
    mean_oscillations_contribution_self = np.zeros((n_pcs,5))
    mean_oscillations_contribution_other = np.zeros((n_pcs,5))
    
    mean_gamma_contribution_self[:,0] = np.arange(n_pcs)
    mean_gamma_contribution_other[:,0] = np.arange(n_pcs)
    mean_oscillations_contribution_self[:,0] = np.arange(n_pcs)
    mean_oscillations_contribution_other[:,0] = np.arange(n_pcs)
    
    for alpha in range(n_pat):
    
        eigenvalues[alpha] = np.loadtxt(in_location+"/eigenvalues_Sigma_"+str(patterns[alpha]))
        eigenvectors[alpha] = np.loadtxt(in_location+"/eigenvectors_Sigma_"+str(patterns[alpha]))
    
    done_self = 0
    done_other = 0
    
    av_eival_dist = np.average(eigenvalues, axis =0)
    np.savetxt(out_location+"/av_eival_dist", av_eival_dist)
        
    for alpha in range(n_pat):
        
        activity_from = patterns[alpha]
        
        total_power = np.sum(eigenvalues[alpha])
        
        for beta in range(n_pat):
            
            PCs_from = patterns[beta]
            
            for pc in select_PCs:
                
                file_name = ("spectrum_pat_"+ str(activity_from) 
                            + "_in_PC_"+str(pc) 
                            +"_of_pat_"+str(PCs_from))
                
                spectrum = np.loadtxt(in_location+"/" + file_name)
                freqs = len(spectrum[:,0])
                total_power_pc = 0.5*np.sum(spectrum[:,1])/(freqs**2)
                
                spectrum = spectrum[1:]
                spectrum = spectrum[spectrum[:,0]<=cut_freq]
                
                x = np.log(spectrum[:,0])
                xx = np.linspace(np.amin(x),np.amax(x), n_points)
                smoothed = smoothen(x,spectrum[:,1],xx)
                                                   
                Pow = np.empty([n_points,2]) 
                Pow[:,0] = np.exp(xx)
                Pow[:,1] = smoothed
                
                selected_indexes_Pow = np.logical_or(
                                        np.logical_and(range_1[0]<Pow[:,0],Pow[:,0]<range_1[1]),
                                        np.logical_and(range_2[0]<Pow[:,0],Pow[:,0]<range_2[1]))
                
                Pow_reduced = np.log(Pow[selected_indexes_Pow])
                
                spectrum_no_gamma = np.log(np.copy(spectrum))
                selected_indexes = np.logical_and(range_1[0]<spectrum[:,0],spectrum[:,0]<range_2[1])
                
                f2 = scipy.interpolate.interp1d(Pow_reduced[:,0],Pow_reduced[:,1])
                spectrum_no_gamma[selected_indexes,1] = f2(spectrum_no_gamma[selected_indexes,0])
                spectrum_no_gamma = np.exp(spectrum_no_gamma)
                
                power_dif_oscillations = 0.5*np.sum(spectrum[:,1]-spectrum_no_gamma[:,1])/(freqs**2)
                
                x_gamma_band = np.logical_and(gamma_range[0]<=spectrum[:,0],
                                                spectrum[:,0]<=gamma_range[1])
                
                power_dif_gamma = 0.5*np.sum(spectrum[x_gamma_band,1]
                                        -spectrum_no_gamma[x_gamma_band,1])/(freqs**2)
                
                alignment = np.abs(np.dot(eigenvectors[alpha,:,pc],eigenvectors[beta,:,pc]))
                
                gamma_contribution[pc,0] = pc
                gamma_contribution[pc,1] = total_power_pc
                gamma_contribution[pc,2] = power_dif_oscillations
                gamma_contribution[pc,3] = power_dif_gamma
                gamma_contribution[pc,4] = power_dif_oscillations/total_power_pc
                gamma_contribution[pc,5] = power_dif_gamma/total_power_pc
                gamma_contribution[pc,6] = power_dif_oscillations/total_power
                gamma_contribution[pc,7] = power_dif_gamma/total_power
                gamma_contribution[pc,8] = alignment
                
                if (alpha == beta):
                    gamma_contribution_self_all[done_self:done_self+1] = gamma_contribution[pc]
                    mean_gamma_contribution_self[pc,1] += gamma_contribution[pc,5]
                    mean_gamma_contribution_self[pc,2] += gamma_contribution[pc,5]**2
                    mean_gamma_contribution_self[pc,3] += gamma_contribution[pc,7]
                    mean_gamma_contribution_self[pc,4] += gamma_contribution[pc,7]**2
                    mean_oscillations_contribution_self[pc,1] += gamma_contribution[pc,4]
                    mean_oscillations_contribution_self[pc,2] += gamma_contribution[pc,4]**2
                    mean_oscillations_contribution_self[pc,3] += gamma_contribution[pc,6]
                    mean_oscillations_contribution_self[pc,4] += gamma_contribution[pc,6]**2
                    done_self += 1
                else:    
                    gamma_contribution_other_all[done_other:done_other+1] = gamma_contribution[pc]
                    mean_gamma_contribution_other[pc,1] += gamma_contribution[pc,5]
                    mean_gamma_contribution_other[pc,2] += gamma_contribution[pc,5]**2
                    mean_gamma_contribution_other[pc,3] += gamma_contribution[pc,7]
                    mean_gamma_contribution_other[pc,4] += gamma_contribution[pc,7]**2
                    mean_oscillations_contribution_other[pc,1] += gamma_contribution[pc,4]
                    mean_oscillations_contribution_other[pc,2] += gamma_contribution[pc,4]**2
                    mean_oscillations_contribution_other[pc,3] += gamma_contribution[pc,6]
                    mean_oscillations_contribution_other[pc,4] += gamma_contribution[pc,6]**2
                    done_other += 1
                
                np.savetxt(out_location+"/"+file_name+"_original",spectrum)
                
                np.savetxt(out_location+"/"+file_name+"_smooth",Pow)
                
                np.savetxt(out_location+"/"+file_name+"_reduced",Pow_reduced)
                
                np.savetxt(out_location+"/"+file_name+"_no_gamma",spectrum_no_gamma)
                
                
            np.savetxt(out_location+"/gamma_contribution_pat_"+ str(activity_from) 
                        + "_in_pcs_of_pat_"+str(PCs_from), gamma_contribution)
        
    mean_gamma_contribution_self[:,1:] /= 1.0*n_pat
    mean_gamma_contribution_self[:,2] = np.sqrt(mean_gamma_contribution_self[:,2] - mean_gamma_contribution_self[:,1]**2)
    mean_gamma_contribution_self[:,4] = np.sqrt(mean_gamma_contribution_self[:,4] - mean_gamma_contribution_self[:,3]**2)
    
    mean_gamma_contribution_other[:,1:] /= 1.0*n_pat*(n_pat-1)
    mean_gamma_contribution_other[:,2] = np.sqrt(mean_gamma_contribution_other[:,2] - mean_gamma_contribution_other[:,1]**2)
    mean_gamma_contribution_other[:,4] = np.sqrt(mean_gamma_contribution_other[:,4] - mean_gamma_contribution_other[:,3]**2)
    
    mean_oscillations_contribution_self[:,1:] /= 1.0*n_pat
    mean_oscillations_contribution_self[:,2] = np.sqrt(mean_oscillations_contribution_self[:,2] - mean_oscillations_contribution_self[:,1]**2)
    mean_oscillations_contribution_self[:,4] = np.sqrt(mean_oscillations_contribution_self[:,4] - mean_oscillations_contribution_self[:,3]**2)
    
    
    mean_oscillations_contribution_other[:,1:] /= 1.0*n_pat*(n_pat-1)
    mean_oscillations_contribution_other[:,2] = np.sqrt(mean_oscillations_contribution_other[:,2] - mean_oscillations_contribution_other[:,1]**2)
    mean_oscillations_contribution_other[:,4] = np.sqrt(mean_oscillations_contribution_other[:,4] - mean_oscillations_contribution_other[:,3]**2)
    
    np.savetxt(out_location+"/gamma_contribution_self_all", gamma_contribution_self_all)
    np.savetxt(out_location+"/gamma_contribution_other_all", gamma_contribution_other_all)
    
    np.savetxt(out_location+"/mean_gamma_contribution_self", mean_gamma_contribution_self)
    np.savetxt(out_location+"/mean_gamma_contribution_other", mean_gamma_contribution_other)
    
    np.savetxt(out_location+"/mean_oscillations_contribution_self", mean_oscillations_contribution_self)
    np.savetxt(out_location+"/mean_oscillations_contribution_other", mean_oscillations_contribution_other)
    
    
# Call Main
if __name__ == "__main__":
    main()
