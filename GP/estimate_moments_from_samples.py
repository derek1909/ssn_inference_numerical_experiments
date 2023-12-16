import numpy as np
import methods as mt
import os

def SKL_2_Gaussians_1d(mu_1,mu_2,sigma2_1,sigma2_2):
    sigma2_1_inv = 1.0/sigma2_1
    sigma2_2_inv = 1.0/sigma2_2
    sigma2_inv = 0.5 * (sigma2_1_inv + sigma2_2_inv)
    SKL_mean = 0.5*((mu_1-mu_2)**2.0)*sigma2_inv
    SKL_cov = 0.25*(sigma2_2_inv * sigma2_1 + sigma2_1_inv * sigma2_2) - 0.5
    SKL_tot = SKL_mean + SKL_cov
    return (SKL_mean,SKL_cov,SKL_tot)


############
#   Main   #
############

def main():
    
    out_location = "SKL_numerical"
                
    if not os.path.exists(out_location):
        os.makedirs(out_location) 
    
    n = 2000
    spacing = 0.5e-3 # in s
    T_max_samples = (n-1) * spacing
    
    mean_GP = 0.0
    var_GP = 4.0
    
    n_draws = 100000 # number of draws from the distribution
    f_gamma = 40 # in Hz
    all_samples_no_gamma = np.loadtxt("samples_no_osc/samples")
    all_samples_gamma = np.loadtxt("samples_osc/samples")
    all_samples_gamma_strong = np.loadtxt("samples_strong_osc/samples")
    all_samples_exp = np.loadtxt("samples_exp/samples")    
    
       
    T_array = np.linspace(start = 0.0, stop = T_max_samples, num=n, endpoint=True)    

    mu_array = np.empty([n,5])
    var_array = np.empty([n,5])
    SKL_mean_array = np.empty([n,5])
    SKL_var_array = np.empty([n,5])  
    SKL_tot_array = np.empty([n,5])    
    
    mu_array[:,0] = T_array
    mu_array[:,1:] = mean_GP
    var_array[:,0] = T_array
    SKL_mean_array[:,0] = T_array
    SKL_var_array[:,0] = T_array
    SKL_tot_array[:,0] = T_array    
    
    for i in range(n):
        T = T_array[i]        
        if (i%10==0):
            print("T = ", T*1000, "ms")
        av_mean_no_gamma = 0.0
        av_var_no_gamma = 0.0
        av_SKL_mean_no_gamma = 0.0
        av_SKL_var_no_gamma = 0.0
        av_SKL_tot_no_gamma = 0.0

        av_mean_gamma = 0.0
        av_var_gamma = 0.0
        av_SKL_mean_gamma = 0.0
        av_SKL_var_gamma = 0.0
        av_SKL_tot_gamma = 0.0
        
        av_mean_gamma_strong = 0.0
        av_var_gamma_strong = 0.0
        av_SKL_mean_gamma_strong = 0.0
        av_SKL_var_gamma_strong = 0.0
        av_SKL_tot_gamma_strong = 0.0

        av_mean_exp = 0.0
        av_var_exp = 0.0
        av_SKL_mean_exp = 0.0
        av_SKL_var_exp = 0.0
        av_SKL_tot_exp = 0.0
        
        for draw in range(n_draws):
            samp_no_gamma = all_samples_no_gamma[:i,1+draw]
            samp_gamma = all_samples_gamma[:i,1+draw]
            samp_gamma_strong = all_samples_gamma_strong[:i,1+draw]
            samp_exp = all_samples_exp[:i,1+draw]
            
            mean_no_gamma = np.mean(samp_no_gamma)
            var_no_gamma = np.var(samp_no_gamma)
            (SKL_mean_no_gamma, SKL_var_no_gamma, SKL_tot_no_gamma) = SKL_2_Gaussians_1d(
                                                                mean_no_gamma,mean_GP,
                                                                var_no_gamma,var_GP)
            av_mean_no_gamma += mean_no_gamma
            av_var_no_gamma += var_no_gamma
            av_SKL_mean_no_gamma += SKL_mean_no_gamma
            av_SKL_var_no_gamma += SKL_var_no_gamma
            av_SKL_tot_no_gamma += SKL_tot_no_gamma

            mean_gamma = np.mean(samp_gamma)
            var_gamma = np.var(samp_gamma)
            (SKL_mean_gamma, SKL_var_gamma, SKL_tot_gamma)  = SKL_2_Gaussians_1d(
                                                            mean_gamma,mean_GP,
                                                            var_gamma,var_GP)
            av_mean_gamma += mean_gamma
            av_var_gamma += var_gamma
            av_SKL_mean_gamma += SKL_mean_gamma
            av_SKL_var_gamma += SKL_var_gamma
            av_SKL_tot_gamma += SKL_tot_gamma

            mean_gamma_strong = np.mean(samp_gamma_strong)
            var_gamma_strong = np.var(samp_gamma_strong)
            (SKL_mean_gamma_strong, SKL_var_gamma_strong, SKL_tot_gamma_strong) = SKL_2_Gaussians_1d(
                                                                        mean_gamma_strong,mean_GP,
                                                                        var_gamma_strong,var_GP)
            av_mean_gamma_strong += mean_gamma_strong
            av_var_gamma_strong += var_gamma_strong
            av_SKL_mean_gamma_strong += SKL_mean_gamma_strong
            av_SKL_var_gamma_strong += SKL_var_gamma_strong
            av_SKL_tot_gamma_strong += SKL_tot_gamma_strong

            mean_exp = np.mean(samp_exp)
            var_exp = np.var(samp_exp)
            (SKL_mean_exp, SKL_var_exp, SKL_tot_exp)  = SKL_2_Gaussians_1d(
                                                            mean_exp,mean_GP,
                                                            var_exp,var_GP)
            av_mean_exp += mean_exp
            av_var_exp += var_exp
            av_SKL_mean_exp += SKL_mean_exp
            av_SKL_var_exp += SKL_var_exp
            av_SKL_tot_exp += SKL_tot_exp

        av_mean_no_gamma /= n_draws
        av_var_no_gamma /= n_draws
        av_SKL_mean_no_gamma /= n_draws
        av_SKL_var_no_gamma /= n_draws
        av_SKL_tot_no_gamma /= n_draws
        
        av_mean_gamma /= n_draws
        av_var_gamma /= n_draws
        av_SKL_mean_gamma /= n_draws
        av_SKL_var_gamma /= n_draws
        av_SKL_tot_gamma /= n_draws

        av_mean_gamma_strong /= n_draws
        av_var_gamma_strong /= n_draws
        av_SKL_mean_gamma_strong /= n_draws
        av_SKL_var_gamma_strong /= n_draws
        av_SKL_tot_gamma_strong /= n_draws

        av_mean_exp /= n_draws
        av_var_exp /= n_draws
        av_SKL_mean_exp /= n_draws
        av_SKL_var_exp /= n_draws
        av_SKL_tot_exp /= n_draws
   
        mu_array[i,1] = av_mean_no_gamma
        mu_array[i,2] = av_mean_gamma
        mu_array[i,3] = av_mean_gamma_strong
        mu_array[i,4] = av_mean_exp

        var_array[i,1] = av_var_no_gamma
        var_array[i,2] = av_var_gamma
        var_array[i,3] = av_var_gamma_strong
        var_array[i,4] = av_var_exp
                                    
        SKL_mean_array[i,1] = av_SKL_mean_no_gamma
        SKL_mean_array[i,2] = av_SKL_mean_gamma
        SKL_mean_array[i,3] = av_SKL_mean_gamma_strong
        SKL_mean_array[i,4] = av_SKL_mean_exp

        SKL_var_array[i,1] = av_SKL_var_no_gamma
        SKL_var_array[i,2] = av_SKL_var_gamma
        SKL_var_array[i,3] = av_SKL_var_gamma_strong
        SKL_var_array[i,4] = av_SKL_var_exp

        SKL_tot_array[i,1] = av_SKL_tot_no_gamma
        SKL_tot_array[i,2] = av_SKL_tot_gamma
        SKL_tot_array[i,3] = av_SKL_tot_gamma_strong
        SKL_tot_array[i,4] = av_SKL_tot_exp

    np.savetxt(out_location+"/av_mean_vs_T", mu_array)
    np.savetxt(out_location+"/av_var_vs_T", var_array)
    np.savetxt(out_location+"/av_SKL_mean_vs_T", SKL_mean_array)
    np.savetxt(out_location+"/av_SKL_cov_vs_T", SKL_var_array)
    np.savetxt(out_location+"/av_SKL_vs_T", SKL_tot_array)
    
    new_indexes = (np.logspace(np.log10(1),np.log10(n),
                    num=100,endpoint=False,base=10.0)).astype(np.int32)
    
    SKL_tot_array_subsamp = SKL_tot_array[new_indexes]

    np.savetxt(out_location+"/av_SKL_vs_T_log_subsamp", SKL_tot_array_subsamp)
      
    
                   
# Call Main
if __name__ == "__main__":
    main()  
