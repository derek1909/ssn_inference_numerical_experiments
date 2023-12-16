import numpy as np
import scipy as sp
import methods as mt
from parameters import *

import datetime, os

def draw_ellipses(mean,eivals,eivecs,n_points):
    t = np.linspace(0,2*np.pi,num= n_points)
    
    ellipse = ( np.outer(np.sqrt(eivals[0])*np.cos(t),eivecs[:,0]) 
                +  np.outer(np.sqrt(eivals[1])*np.sin(t),eivecs[:,1]))
    
    mean = np.expand_dims(mean, axis =0)
    
    return (mean + ellipse, mean + 2.0 * ellipse, mean + 3.0 * ellipse)

def trajectories_and_ellipsoids(W,Sigma_eta,selected_neurons,keys_for_Sigma,
                                patterns,in_location,out_location,train=True):
    
    N_pat = len(patterns)
    n_points_ellipse = 100
    sample_size = 10000
    t_bet_samp = 10 * tau_e
    steps_bet_samp = int(t_bet_samp/dt)  
    
    sample_size_short = 1000
    t_bet_samp_short = 0.2 * tau_e
    steps_bet_samp_short = int(t_bet_samp_short/dt) 
    
    print("Start time : ", datetime.datetime.now())
           
    for pat in range(N_pat):
        alpha = patterns[pat]
        if train:
            print("trainset")
            h = np.loadtxt(in_location+"/h_true_"+str(alpha)+"_learn") 
            mu_0 = np.loadtxt(in_location+"/mu_learn_"+str(alpha))
            Sigma_0 = np.loadtxt(in_location+"/sigma_learn_"+str(alpha))
            mu_GSM = np.loadtxt(in_location+"/mu"+str(alpha))
            Sigma_GSM = np.loadtxt(in_location+"/sigma"+str(alpha))
        else:
            print("testset")
            h = np.loadtxt(in_location+"/h"+str(alpha)) 
            mu_0 = np.loadtxt(in_location+"/mu_SSN_"+str(alpha))
            Sigma_0 = np.loadtxt(in_location+"/Sigma_SSN_"+str(alpha))
            mu_GSM = np.loadtxt(in_location+"/mu_GSM_"+str(alpha))
            Sigma_GSM = np.loadtxt(in_location+"/Sigma_GSM_"+str(alpha))
        
        u0 = np.random.multivariate_normal(mean = mu_0, cov = Sigma_0)
        eta0 = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma_eta)
        print("evolving pattern " + str(alpha))
        u, eta = mt.network_evolution(W,h,u0,Sigma_eta, eta = eta0)
        print("sampling moments")
        (u_samples,_,u_new,eta_new) = mt.network_sample(W,h,u,eta,sample_size,steps_bet_samp,Sigma_eta)
        print("short trajectory")
        (u_samples_short,_,_,_) = mt.network_sample(W,h,u_new,eta_new,
                                    sample_size_short,steps_bet_samp_short,Sigma_eta)
        
        
        mu_final = np.mean(u_samples, axis=0)
        Sigma_final = np.cov(u_samples, rowvar=False)
        
        u_samples_2d = u_samples[0:1000:10,selected_neurons]
        u_samples_2d_short = u_samples_short[:,selected_neurons]
        
        mu_SSN_2d = mu_final[selected_neurons]
        mu_GSM_2d = mu_GSM[selected_neurons]
        
        Sigma_SSN_2d = Sigma_final[keys_for_Sigma]
        Sigma_GSM_2d = Sigma_GSM[keys_for_Sigma]
        
        eivals_SSN, eivecs_SSN = np.linalg.eigh(Sigma_SSN_2d)
        eivals_GSM, eivecs_GSM = np.linalg.eigh(Sigma_GSM_2d)
                
        idx_SSN = eivals_SSN.argsort()[::-1] 
        eigenvalues_SSN= np.copy(eivals_SSN[idx_SSN])
        eigenvectors_SSN = np.copy(eivecs_SSN[:,idx_SSN])
        
        idx_GSM = eivals_GSM.argsort()[::-1] 
        eigenvalues_GSM = np.copy(eivals_GSM[idx_GSM])
        eigenvectors_GSM = np.copy(eivecs_GSM[:,idx_GSM])
        
        (ellipse_SSN_1sd,ellipse_SSN_2sd, ellipse_SSN_3sd) = draw_ellipses(
                                mu_SSN_2d,eigenvalues_SSN,eigenvectors_SSN,n_points_ellipse)
        
        (ellipse_GSM_1sd,ellipse_GSM_2sd, ellipse_GSM_3sd) = draw_ellipses(
                                mu_GSM_2d,eigenvalues_GSM,eigenvectors_GSM,n_points_ellipse)
        
        np.savetxt(out_location+"/samples_SSN_"+str(alpha),u_samples_2d)
        np.savetxt(out_location+"/samples_SSN_short_"+str(alpha),u_samples_2d_short)
        np.savetxt(out_location+"/ellipse_SSN_1sd_"+str(alpha),ellipse_SSN_1sd)
        np.savetxt(out_location+"/ellipse_GSM_1sd_"+str(alpha),ellipse_GSM_1sd)
        np.savetxt(out_location+"/ellipse_SSN_2sd_"+str(alpha),ellipse_SSN_2sd)
        np.savetxt(out_location+"/ellipse_GSM_2sd_"+str(alpha),ellipse_GSM_2sd)
        np.savetxt(out_location+"/ellipse_SSN_3sd_"+str(alpha),ellipse_SSN_3sd)
        np.savetxt(out_location+"/ellipse_GSM_3sd_"+str(alpha),ellipse_GSM_3sd)
        
    print("End time : ", datetime.datetime.now())


############
#   Main   #
############

def main():

    in_location = "parameter_files"
    
    selected_neurons = np.array([13,20]) #np.array([24,18])
    keys_for_Sigma = np.meshgrid(selected_neurons,selected_neurons)
    
    #----------------------------------------------#
    #   We import W and the noise covariance   #
    #----------------------------------------------#

    Sigma_eta = np.loadtxt(in_location+"/sigma_eta_learn")
    
    W = np.loadtxt(in_location+"/w_learn")
       

    # train
    
    in_location_patterns = in_location
    
    out_location = "results/samples_and_ellipsoids_in_2d/train"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)  
       
    patterns = np.array([0,1,2,3,4])
    
    trajectories_and_ellipsoids(W,Sigma_eta,selected_neurons,keys_for_Sigma,
                             patterns,in_location_patterns,out_location,train=True)
    
    
    # test
    
    in_location_patterns = in_location + "/div_norm_data/z_from_gamma/test"
       
    out_location = in_location + "/samples_and_ellipsoids_in_2d/test"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)  
    
    patterns = np.array([18,40,42,44,61])
    
    trajectories_and_ellipsoids(W,Sigma_eta,selected_neurons,keys_for_Sigma,
                             patterns,in_location_patterns,out_location,train=False)

# Call Main
if __name__ == "__main__":
    main()
