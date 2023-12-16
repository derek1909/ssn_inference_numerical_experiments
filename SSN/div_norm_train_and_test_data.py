import numpy as np
import scipy as sp
import methods as mt
from parameters import *

import datetime, os

def np_th(u):
    return np.maximum(u,0)
    
def get_true_h_from_x_proj(x_proj,h_scale,input_scaling,input_baseline,input_pow):
    # h = h_scale * x_proj
    # true_h = input_scaling*(input_baseline + h)**input_pow
    h = h_scale*x_proj
    true_h = input_scaling*((np_th(input_baseline + h))**input_pow)
    return np.concatenate([true_h,true_h])

############
#   Main   #
############

def main():
    
    
    in_location = "../GSM/generalization_data/no_noise/results_GSM_nl"
    net_params_location = "parameter_files"
    
    out_location_train = net_params_location + "/div_norm_data/z_from_gamma/train"
    out_location_test = net_params_location + "/div_norm_data/z_from_gamma/test"
       
    epsilon = 1.0E-12
    baseline = 0.0
    
    if not os.path.exists(out_location_train):
        os.makedirs(out_location_train)
        
    if not os.path.exists(out_location_test):
        os.makedirs(out_location_test)
    
    n_points_train = 100 * N_pat
    n_points_test = 100 * N_pat
    
    
    A = np.loadtxt(net_params_location+"/A")
    A_T = A.T
    h_scale = 1.0/15.0
    input_scaling = np.loadtxt(net_params_location+"/input_scaling_learn")
    input_baseline = np.loadtxt(net_params_location+"/input_baseline_learn")
    input_pow = np.loadtxt(net_params_location+"/input_nl_pow_learn")
    
    print("Working with " + str(n_points_train +n_points_test) + " patterns")
      
    #----------------------------------------------#
    #   We import W, h, and the noise covariance   #
    #----------------------------------------------#

    Sigma_eta = np.loadtxt(net_params_location+"/sigma_eta_learn")
        
    W = np.loadtxt(net_params_location+"/w_learn")
    
    #------------------------------#
    #   Evolution of the network   #
    #------------------------------#
           
    sample_size = 5000
    t_bet_samp = 10 * tau_e
    steps_bet_samp = int(t_bet_samp/dt)  
    
    
    print("Start time : ", datetime.datetime.now())
    
    for alpha in range(n_points_test):
        alpha_test = alpha
        print("test point ", alpha)
        #h = np.loadtxt(in_location+"/h_true_"+str(alpha_test))
        z = np.loadtxt(in_location+"/z_"+str(alpha_test))
        x = np.loadtxt(in_location+"/x_"+str(alpha_test))
        h_lin = h_scale*np.dot(A_T,x)
        h = get_true_h_from_x_proj(np.dot(A_T,x),h_scale,input_scaling,input_baseline,input_pow)
        mu_GSM = (baseline + np.loadtxt(in_location+"/mu_true_z_"+str(alpha_test)))
        Sigma_GSM = np.loadtxt(in_location+"/Sigma_true_z_"+str(alpha_test))
        diag_Sigma_GSM =  np.diag(Sigma_GSM)
        std_GSM = np.loadtxt(in_location+"/std_true_z_"+str(alpha_test))
        
        mu0 = np.concatenate([mu_GSM,mu_GSM],axis = 0)
        Sigma_A = np.concatenate([Sigma_GSM,0.5*Sigma_GSM],axis = 0)
        Sigma_B = np.concatenate([0.5*Sigma_GSM,Sigma_GSM],axis = 0)
        Sigma0 = np.concatenate([Sigma_A,Sigma_B],axis = 1)
        
        u0 = np.random.multivariate_normal(mean = mu0, cov = Sigma0)
        eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
        print("evolving pattern " + str(alpha))
        u, eta = mt.network_evolution(W,h,u0,Sigma_eta, eta = eta0)
        print("sampling moments")

        (samples,_,_,_) = mt.network_sample(W,h,u,eta,sample_size,steps_bet_samp,Sigma_eta)
        mu_SSN = np.mean(samples, axis=0)
        Sigma_SSN = np.cov(samples, rowvar=False)
        diag_Sigma_SSN =  np.diag(Sigma_SSN)
        std_SSN = np.sqrt(diag_Sigma_SSN)
        
        nu_SSN = mt.get_nu(mu_SSN, diag_Sigma_SSN)
        Lambda_SSN = mt.get_Lambda(mu_SSN,nu_SSN,Sigma_SSN,diag_Sigma_SSN)

        lambda_std_SSN =  np.sqrt(np.diag(Lambda_SSN))
        
        nu_GSM = mt.get_nu(mu_GSM, diag_Sigma_GSM)
        Lambda_GSM = mt.get_Lambda(mu_GSM,nu_GSM,Sigma_GSM,diag_Sigma_GSM)
        lambda_std_GSM =  np.sqrt(np.diag(Lambda_GSM))
        
        np.savetxt(out_location_test+"/mu_GSM_"+str(alpha),mu_GSM)
        np.savetxt(out_location_test+"/Sigma_GSM_"+str(alpha),Sigma_GSM)
        np.savetxt(out_location_test+"/std_GSM_"+str(alpha),std_GSM)
        
        np.savetxt(out_location_test+"/z_"+str(alpha),np.expand_dims(z, axis = 0))
        np.savetxt(out_location_test+"/x_"+str(alpha),x)
        np.savetxt(out_location_test+"/h_lin_"+str(alpha),np.concatenate([h_lin,h_lin]))
        np.savetxt(out_location_test+"/h"+str(alpha),h)
        np.savetxt(out_location_test+"/mu_SSN_"+str(alpha),mu_SSN)
        np.savetxt(out_location_test+"/Sigma_SSN_"+str(alpha),Sigma_SSN)
        np.savetxt(out_location_test+"/std_SSN_"+str(alpha),std_SSN)
        
        np.savetxt(out_location_test+"/nu_SSN_"+str(alpha),nu_SSN)
        np.savetxt(out_location_test+"/Lambda_SSN_"+str(alpha),Lambda_SSN)
        np.savetxt(out_location_test+"/lambda_std_SSN_"+str(alpha),lambda_std_SSN)
        
        np.savetxt(out_location_test+"/nu_GSM_"+str(alpha),nu_GSM)
        np.savetxt(out_location_test+"/Lambda_GSM_"+str(alpha),Lambda_GSM)
        np.savetxt(out_location_test+"/lambda_std_GSM_"+str(alpha),lambda_std_GSM)
    
           
    for alpha in range(n_points_train):
        alpha_train = alpha + n_points_test
        print("train point ", alpha)
        #h = np.loadtxt(in_location+"/h_true_"+str(alpha))
        z = np.loadtxt(in_location+"/z_"+str(alpha_train))
        x = np.loadtxt(in_location+"/x_"+str(alpha_train))
        h_lin = h_scale*np.dot(A_T,x)
        h = get_true_h_from_x_proj(np.dot(A_T,x),h_scale,input_scaling,input_baseline,input_pow)
        mu_GSM = (baseline + np.loadtxt(in_location+"/mu_true_z_"+str(alpha_train)))
        Sigma_GSM = np.loadtxt(in_location+"/Sigma_true_z_"+str(alpha_train))
        diag_Sigma_GSM =  np.diag(Sigma_GSM)
        std_GSM = np.loadtxt(in_location+"/std_true_z_"+str(alpha_train))
        
        mu0 = np.concatenate([mu_GSM,mu_GSM],axis = 0)
        Sigma_A = np.concatenate([Sigma_GSM,0.5*Sigma_GSM],axis = 0)
        Sigma_B = np.concatenate([0.5*Sigma_GSM,Sigma_GSM],axis = 0)
        Sigma0 = np.concatenate([Sigma_A,Sigma_B],axis = 1)
        
        u0 = np.random.multivariate_normal(mean = mu0, cov = Sigma0)
        eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
        print("evolving pattern " + str(alpha))
        u, eta = mt.network_evolution(W,h,u0,Sigma_eta, eta = eta0)
        print("sampling moments")
        (samples,_,_,_) = mt.network_sample(W,h,u,eta,sample_size,steps_bet_samp,Sigma_eta)
        mu_SSN = np.mean(samples, axis=0)
        Sigma_SSN = np.cov(samples, rowvar=False)
        diag_Sigma_SSN =  np.diag(Sigma_SSN)
        std_SSN = np.sqrt(diag_Sigma_SSN)
        
        nu_SSN = mt.get_nu(mu_SSN, diag_Sigma_SSN)
        Lambda_SSN = mt.get_Lambda(mu_SSN,nu_SSN,Sigma_SSN,diag_Sigma_SSN)
        lambda_std_SSN =  np.sqrt(np.diag(Lambda_SSN))
        
        nu_GSM = mt.get_nu(mu_GSM, diag_Sigma_GSM)
        Lambda_GSM = mt.get_Lambda(mu_GSM,nu_GSM,Sigma_GSM,diag_Sigma_GSM)
        lambda_std_GSM =  np.sqrt(np.diag(Lambda_GSM))
                       
        np.savetxt(out_location_train+"/mu_GSM_"+str(alpha),mu_GSM)
        np.savetxt(out_location_train+"/Sigma_GSM_"+str(alpha),Sigma_GSM)
        np.savetxt(out_location_train+"/std_GSM_"+str(alpha),std_GSM)
        
        np.savetxt(out_location_train+"/z_"+str(alpha),np.expand_dims(z, axis = 0))
        np.savetxt(out_location_train+"/x_"+str(alpha),x)
        np.savetxt(out_location_train+"/h_lin_"+str(alpha),np.concatenate([h_lin,h_lin]))
        np.savetxt(out_location_train+"/h"+str(alpha),h)
        np.savetxt(out_location_train+"/mu_SSN_"+str(alpha),mu_SSN)
        np.savetxt(out_location_train+"/Sigma_SSN_"+str(alpha),Sigma_SSN)
        np.savetxt(out_location_train+"/std_SSN_"+str(alpha),std_SSN)
        
        np.savetxt(out_location_train+"/nu_SSN_"+str(alpha),nu_SSN)
        np.savetxt(out_location_train+"/Lambda_SSN_"+str(alpha),Lambda_SSN)
        np.savetxt(out_location_train+"/lambda_std_SSN_"+str(alpha),lambda_std_SSN)
        
        np.savetxt(out_location_train+"/nu_GSM_"+str(alpha),nu_GSM)
        np.savetxt(out_location_train+"/Lambda_GSM_"+str(alpha),Lambda_GSM)
        np.savetxt(out_location_train+"/lambda_std_GSM_"+str(alpha),lambda_std_GSM)
     
    
     
    print("End time : ", datetime.datetime.now())
        
       

# Call Main
if __name__ == "__main__":
    main()
