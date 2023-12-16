import numpy as np
from parameters import *
import os

def SKL_2_Gaussians_1d(mu_1,mu_2,sigma2_1,sigma2_2):
    sigma2_1_inv = 1.0/sigma2_1
    sigma2_2_inv = 1.0/sigma2_2
    sigma2_inv = 0.5 * (sigma2_1_inv + sigma2_2_inv)
    SKL_mean = 0.5*((mu_1-mu_2)**2.0)*sigma2_inv
    SKL_cov = 0.25*(sigma2_2_inv * sigma2_1 + sigma2_1_inv * sigma2_2) - 0.5
    SKL_tot = SKL_mean + SKL_cov
    return SKL_tot

def covariance(delta_t, s2, tau, strong, f = 0):
    if strong:
        c1 = 0.3
        c2 = 0.7
    else:
        c1 = 0.8
        c2 = 0.2
    if (f==0):
        return s2 * (1 + delta_t/tau) * np.exp(-delta_t/tau)
    else:
        return s2 * (c1+c2*np.cos(delta_t*two_pi*f)) * (1 + delta_t/tau) * np.exp(-delta_t/tau)


############
#   Main   #
############

def main():
    
    oscillations = True
    strong = True
    
    out_location = "SKL_analytic"
                    
    if not os.path.exists(out_location):
        os.makedirs(out_location) 
    
    T_min = 1.0e-3 # 1ms
    T_max = 1.0 # 1s

    n_Ts = 300

    f_gamma = 40 # in Hz

    mean_GP = 0.0
    var_GP = 4.0
    tau_GP = 20.0e-3 # in s   

    T_array = np.logspace(np.log10(T_min),np.log10(T_max),num=n_Ts,endpoint=True,base=10.0)

    mu_array = np.empty([n_Ts,4])
    mse_array = np.empty([n_Ts,4])
    var_array = np.empty([n_Ts,4])
    
    SKL_array = np.empty([n_Ts,4])
    SKL_array_corrected = np.empty([n_Ts,4])    
    
    n_points = 10000
    
    mu_array[:,0] = T_array
    mu_array[:,1:] = mean_GP
    mse_array[:,0] = T_array
    var_array[:,0] = T_array
    SKL_array[:,0] = T_array
    SKL_array_corrected[:,0] = T_array    

    for i in range(n_Ts):
        T = T_array[i]
        if (i%10==0):
            print("T = ", T*1000, "ms")        
        t_array = np.linspace(0,T,num=n_points,endpoint=True)
        d_array = (np.abs(np.subtract.outer(t_array,t_array))).flatten()
        dt = t_array[1]-t_array[0]
        mse_array[i,1] = np.sum(covariance(d_array, var_GP, tau_GP, False, f = 0)) * (dt/T)**2.0
        mse_array[i,2] = np.sum(covariance(d_array, var_GP, tau_GP, False, f = f_gamma)) * (dt/T)**2.0
        mse_array[i,3] = np.sum(covariance(d_array, var_GP, tau_GP, True, f = f_gamma)) * (dt/T)**2.0
                
        var_array[i,1:] = var_GP - mse_array[i,1:]         
            
        SKL_array[i,1] = SKL_2_Gaussians_1d(mean_GP,mu_array[i,1],var_GP,var_array[i,1])
        SKL_array[i,2] = SKL_2_Gaussians_1d(mean_GP,mu_array[i,2],var_GP,var_array[i,2])
        SKL_array[i,3] = SKL_2_Gaussians_1d(mean_GP,mu_array[i,3],var_GP,var_array[i,3])

        var_1_inv = 1.0/var_GP
        var_2_inv_no_gamma = 1.0/var_array[i,1]
        var_inv_no_gamma = 0.5 * (var_1_inv + var_2_inv_no_gamma)
        
        var_2_inv_gamma = 1.0/var_array[i,2]
        var_inv_gamma = 0.5 * (var_1_inv + var_2_inv_gamma)
        
        var_2_inv_gamma_strong = 1.0/var_array[i,3]
        var_inv_gamma_strong = 0.5 * (var_1_inv + var_2_inv_gamma_strong) 
        
        SKL_array_corrected[i,1] = SKL_array[i,1] + 0.5 * var_inv_no_gamma * mse_array[i,1]
        SKL_array_corrected[i,2] = SKL_array[i,2] + 0.5 * var_inv_gamma * mse_array[i,2]
        SKL_array_corrected[i,3] = SKL_array[i,3] + 0.5 * var_inv_gamma_strong * mse_array[i,3]

    np.savetxt(out_location+"/av_mean_vs_T", mu_array)
    np.savetxt(out_location+"/av_var_vs_T", var_array)
    np.savetxt(out_location+"/mse_of_mean_vs_T", mse_array)
    np.savetxt(out_location+"/av_SKL_vs_T", SKL_array)
    np.savetxt(out_location+"/av_SKL_vs_T_corrected", SKL_array_corrected)
    
                   
# Call Main
if __name__ == "__main__":
    main()  
