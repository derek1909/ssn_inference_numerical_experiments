# This code takes the set of inputs for which the net has been trained,
# generates several noisy versions of them, and compares the average infered result 
# from the GSM, with the average ssn result 

import autograd.numpy as np
import autograd.scipy as sp
import methods as mt
from parameters import *
import datetime
import warnings
warnings.filterwarnings('error')
import os

def rotate(V,n):
    L = len(V)
    V_new = np.empty([L])
    
    for i in range(L):
        V_new[i] = V[(i-n)%L]
    
    
    return V_new
    
def get_x(y,z,A,s_x):
    x_mean = z*np.dot(A,y)
    noise = np.random.normal(scale=s_x, size=len(x_mean))
    return np.add(x_mean,noise)
    
def get_post_mean(z,s_x_2,Sigma_post,A,x):
    mu = (z/s_x_2)*np.dot(Sigma_post,np.dot(A.T,x))
    return mu 



############
#   Main   #
############

def main():
    
    location = "parameter_files"
    out_location_data = "generalized_targets"
    
    if not os.path.exists(out_location_data):
        os.makedirs(out_location_data)

    epsilon = 1.0E-12
    
    # Dist params:
    s_x = 10.0                # Noise of the x process
    s_x_2 = s_x**2
    
    h_scale = 1.0/15.0
    
    A = np.load(location + "/filters.npy")
    #----------------------------------------------#
    #   We import W, h, and the noise covariance   #
    #----------------------------------------------#

    Sigma_eta = np.loadtxt(location+"/sigma_eta_learn")
    
    noise_scale = 1.0
    y = np.empty([N_pat+1,N_cons])
    
    input_scale = np.loadtxt(location+"/input_scaling_learn")
    input_baseline = np.loadtxt(location+"/input_baseline_learn")
    input_power = np.loadtxt(location+"/input_nl_pow_learn")
    
    z_array = np.array([0.0,0.125,0.25,0.5,1.0])
           
    for alpha in range(N_pat):
        y[alpha] = np.loadtxt(location+"/y_"+str(alpha))
        
    
    W = np.loadtxt(location+"/w_learn")
       
    #------------------------------#
    #   Evolution of the moments   #
    #------------------------------#
           
    # We take simple initial conditions for mu and Sigma
    mu_GSM = np.empty([N_pat+1,N_cons])
    Sigma_GSM = np.empty([N_pat+1,N_cons,N_cons])
    
    mu_evolved = np.empty([N_pat+1,N])
    Sigma_evolved = np.empty([N_pat+1,N,N])
    
    mu_comparison = np.empty([N_cons,2])
    std_comparison = np.empty([N_cons,2])
        
    stats = 100
    
    print("Start time : ", datetime.datetime.now())
    
    tot_fail = 0
               
    for alpha in range(N_pat):
        print("Pat : " + str(alpha))
        mu_evolved[alpha] = np.zeros([N])
        Sigma_evolved[alpha] = np.zeros([N,N])
        mu_GSM[alpha] = np.zeros([N_cons])
        Sigma_GSM[alpha] = np.loadtxt(location+"/sigma"+str(alpha))
        
        h_mean = np.zeros([N])
        
        mu_0 = np.loadtxt("results/mu_learn_"+str(alpha))
        Sigma_0 = np.loadtxt("results/sigma_learn_"+str(alpha))
        
        for s in range(stats):
            print("Iter : " + str(s))
            
            while True:
                try:
                    x = get_x(y[alpha],z_array[alpha],A,s_x)
                    h = h_scale*np.dot(A.T,x)
                    h = np.concatenate((h,h))
                    h_true = input_scale * np.power(h + input_baseline, input_power)
                    mu_ssn, Sigma_ssn = mt.evolve_1_pattern(out_location_data,W,h_true,
                                    mu_0,Sigma_0,Sigma_eta,noise_scale)
                    break                
                except:
                    print ("one failed, retrying")
                    tot_fail += 1
                
            
            h_mean += h_true
            
            mu_evolved[alpha] += mu_ssn
            Sigma_evolved[alpha] += Sigma_ssn
            
            mu_gsm = get_post_mean(z_array[alpha],s_x_2,Sigma_GSM[alpha],A,x)
            
            mu_GSM[alpha] += mu_gsm
            
            np.savetxt(out_location_data+"/iters/h_true_"+str(alpha)+"_iter_"+str(s),h_true)
            np.savetxt(out_location_data+"/iters/x_"+str(alpha)+"_iter_"+str(s),x)
            np.savetxt(out_location_data+"/iters/mu_GSM_"+str(alpha)+"_iter_"+str(s),mu_gsm)
            np.savetxt(out_location_data+"/iters/mu_SSN_"+str(alpha)+"_iter_"+str(s),mu_ssn)
            np.savetxt(out_location_data+"/iters/sigma_SSN_"+str(alpha)+"_iter_"+str(s),Sigma_ssn)
            np.savetxt(out_location_data+"/iters/std_SSN_"+str(alpha)+"_iter_"+str(s),np.sqrt(np.diag(Sigma_ssn)))
            
        mu_evolved[alpha] = mu_evolved[alpha]/(1.0*stats)
        Sigma_evolved[alpha] = Sigma_evolved[alpha]/(1.0*stats)
        
        h_mean = h_mean/(1.0*stats)
        
        mu_GSM[alpha] = mu_GSM[alpha]/(1.0*stats)
        
        mu_comparison[:,0] = mu_GSM[alpha]
        mu_comparison[:,1] = mu_evolved[alpha,:N_cons]
        
        std_comparison[:,0] = np.sqrt(np.diag(Sigma_GSM[alpha]))
        std_comparison[:,1] = np.sqrt(np.diag(Sigma_evolved[alpha]))[:N_cons]
        
        np.savetxt(out_location_data+"/h_mean"+str(alpha),h_mean)
        np.savetxt(out_location_data+"/mu_evolved"+str(alpha),mu_evolved[alpha])
        np.savetxt(out_location_data+"/mu_comparison"+str(alpha),mu_comparison)
        np.savetxt(out_location_data+"/sigma_evolved"+str(alpha),Sigma_evolved[alpha])
        np.savetxt(out_location_data+"/std_evolved"+str(alpha),np.sqrt(np.diag(Sigma_evolved[alpha])))
        np.savetxt(out_location_data+"/std_comparison"+str(alpha),std_comparison)
       
    print("End time : ", datetime.datetime.now())
    print(str(tot_fail) + " simulations failed and had to be redone")
    
# Call Main
if __name__ == "__main__":
    main()
