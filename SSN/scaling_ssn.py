# Takes the training set and extends the calculation 
# to other contrast levels

import numpy as np
import scipy as sp
import methods as mt
from parameters import *
import datetime, sys, os


############
#   Main   #
############

def main():
    
    use_full_net = True  # If false, compute scaling using ADF instead
    
    if (use_full_net):
        sample_size = 20000
        t_bet_samp = 10 * tau_e
        steps_bet_samp = int(t_bet_samp/dt)
    
    in_location = "parameter_files"
    out_location = "results/scaling"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
        
    epsilon = 1.0E-12
      
    #----------------------------------------------#
    #   We import W, h, and the noise covariance   #
    #----------------------------------------------#
    
    N_pat = 100
    
    Sigma_eta = np.loadtxt(in_location+"/sigma_eta_learn")
        
    noise_scale = 1.0
    h = np.empty([N_pat,N])
    z = np.empty([N_pat])
    
    input_scaling_learn = np.loadtxt(in_location+"/input_scaling_learn")
    input_baseline_learn = np.loadtxt(in_location+"/input_baseline_learn")
    input_nl_pow_learn = np.loadtxt(in_location+"/input_nl_pow_learn")
    
    h_full_contrast = np.loadtxt(in_location+"/h4")
    
    for alpha in range(N_pat):
        z[alpha] = 2.0*(1.0*alpha/(1.0*(N_pat-1)))
        h_lin= z[alpha] * h_full_contrast
        h[alpha] = input_scaling_learn * np.power((h_lin + input_baseline_learn),input_nl_pow_learn)
    
    W = np.loadtxt(in_location+"/w_learn")
       
    #------------------------------#
    #   Evolution of the moments   #
    #------------------------------#
           
    # We take simple initial conditions for mu and Sigma
    mu_0 = np.zeros(N)
    Sigma_0 = 4.0*np.identity(N)       
              
    mu_final = np.empty([N_pat,N])
    nu_final = np.empty([N_pat,N])
    Sigma_final = np.empty([N_pat,N,N])
    std_final = np.empty([N_pat,N])
    Lambda_final = np.empty([N_pat,N,N])
    std_r_final = np.empty([N_pat,N])
    results = np.empty([N_pat,8])
    
    print("Start time : ", datetime.datetime.now())
    
    for alpha in range(N_pat):
        
        print("evolving pattern" + str(alpha))
        
        if (use_full_net):
            u0 = np.random.multivariate_normal(mean = mu_0, cov = Sigma_0)
            eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
            u, eta = mt.network_evolution(W,h[alpha],u0,Sigma_eta, eta = eta0)
            print("sampling moments")
            (u_samples,_,_,_) = mt.network_sample(W,h[alpha],u,eta,sample_size,steps_bet_samp,Sigma_eta)
            mu_final[alpha] = np.mean(u_samples, axis=0)
            Sigma_final[alpha] = np.cov(u_samples, rowvar=False)
            diag_Sigma = np.diag(Sigma_final[alpha])
            r_samples = mt.get_r(u_samples)
            nu_final[alpha] = np.mean(r_samples, axis=0)
            Lambda_final[alpha] = np.cov(r_samples, rowvar=False)
        else:
            (mu_final[alpha],Sigma_final[alpha],_) = mt.moment_evolution(out_location,W,h,
                                                                 mu_0,Sigma_0,
                                                                 Sigma_eta,noise_scale,
                                                                 alpha)
        
            diag_Sigma = np.diag(Sigma_final[alpha])
            nu_final[alpha] = mt.get_nu(mu_final[alpha], diag_Sigma)
            Lambda_final[alpha] = mt.get_Lambda(mu_final[alpha],nu_final[alpha],
                                            Sigma_final[alpha], diag_Sigma)
        
        diag_Lambda = np.diag(Lambda_final[alpha])
        std_final[alpha] = np.sqrt(diag_Sigma)
        std_r_final[alpha] = np.sqrt(diag_Lambda)
        
        results[alpha][0] = z[alpha]
        results[alpha][1] = np.average(h[alpha,:N_exc])
        results[alpha][2] = np.average(mu_final[alpha,:N_exc])
        results[alpha][3] = np.amax(mu_final[alpha,:N_exc])
        results[alpha][4] = np.average(std_final[alpha,:N_exc])
        results[alpha][5] = np.average(nu_final[alpha,:N_exc])
        results[alpha][6] = np.amax(nu_final[alpha,:N_exc])
        results[alpha][7] = np.average(std_r_final[alpha,:N_exc])
        
        
        if (use_full_net):
            np.savetxt(out_location+"/mu_evolved_net_"+str(alpha),mu_final[alpha])
            np.savetxt(out_location+"/sigma_evolved_net_"+str(alpha),Sigma_final[alpha])
            np.savetxt(out_location+"/std_evolved_net_"+str(alpha),std_final[alpha])
            np.savetxt(out_location+"/nu_evolved_net_"+str(alpha),nu_final[alpha])
            np.savetxt(out_location+"/lambda_evolved_net_"+str(alpha),Lambda_final[alpha])
            np.savetxt(out_location+"/std_r_evolved_net_"+str(alpha),std_r_final[alpha])
            np.savetxt(out_location+"/mu_and_std_vs_contrast_ssn_full_net",results)
            np.savetxt(out_location+"/all_mus_evolved_net",mu_final)
            np.savetxt(out_location+"/all_nus_evolved_net",nu_final)
        else:
            np.savetxt(out_location+"/mu_evolved_"+str(alpha),mu_final[alpha])
            np.savetxt(out_location+"/sigma_evolved_"+str(alpha),Sigma_final[alpha])
            np.savetxt(out_location+"/std_evolved_"+str(alpha),std_final[alpha])
            np.savetxt(out_location+"/nu_evolved_"+str(alpha),nu_final[alpha])
            np.savetxt(out_location+"/lambda_evolved_"+str(alpha),Lambda_final[alpha])
            np.savetxt(out_location+"/std_r_evolved_"+str(alpha),std_r_final[alpha])
            np.savetxt(out_location+"/mu_and_std_vs_contrast_ssn",results)
            np.savetxt(out_location+"/all_mus_evolved",mu_final)
            np.savetxt(out_location+"/all_nus_evolved",nu_final)
    
    print("End time : ", datetime.datetime.now())
    

# Call Main
if __name__ == "__main__":
    main()
