# This code computes the level of oscillatoriness from the autocorrelation function via a parametric fit

import numpy as np
import scipy as sp

import methods as mt
#from parameters import *

import tensorflow as tf
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
from scipy.optimize import least_squares

import datetime, os, gc


def error_TF(x,y):
    return tf.reduce_mean(tf.squared_difference(x,y))


def autocorr_param_numpy(t_array,alpha,tau,tau_n,f_osc):
    two_pi = 2.0 * np.pi
    return (
        ((1-alpha)+alpha*np.cos(t_array*two_pi*f_osc)) * 
        ((tau_n/(tau_n-tau)) * np.exp(-t_array/tau_n) - (tau/(tau_n-tau)) * np.exp(-t_array/tau))
    )
    
def fit_autocorr_TF(autocorr, n_init):
    # Use CPUs
    #config = tf.ConfigProto(
    #    device_count = {'GPU': 0}
    #)
        
    
    # Suppress some level of logs
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow import logging
    logging.set_verbosity(logging.INFO)
    
    params_array = np.empty([n_init,4])
    cost_array = np.empty([n_init])
        
    t_array = tf.constant(autocorr[:,0], dtype=np.float32)
    autocorr_array = tf.constant(autocorr[:,1], dtype=np.float32)
    #tau_n_tf = tf.constant(tau_n, dtype=np.float32)
    two_pi_tf = tf.constant(2.0 * np.pi, dtype=np.float32)
    
    
    param_0 = tf.Variable(tf.random_uniform([1],
                        minval=-1.0,
                        maxval=1.0), 
                        dtype=np.float32)                    
    
    param_1 = tf.Variable(tf.random_uniform([1],
                        minval=0.0,
                        maxval=np.sqrt(40.0e-3)), 
                        dtype=np.float32)
                        
    param_2 = tf.Variable(tf.random_uniform([1],
                        minval=0.0,
                        maxval=np.sqrt(40.0)), 
                        dtype=np.float32)
    
    param_3 = tf.Variable(tf.random_uniform([1],
                        minval=0.0,
                        maxval=np.sqrt(100.0)), 
                        dtype=np.float32)
    
    alpha = 1.0 / (1.0 + tf.exp(param_0))
    tau = param_1 * param_1
    tau_n_tf = param_2 * param_2
    f_osc = 20.0 + param_3 * param_3
    autocorr_array_param = (
        ((1-alpha)+alpha*tf.cos(t_array*two_pi_tf*f_osc)) * 
        ((tau_n_tf/(tau_n_tf-tau)) * tf.exp(-t_array/tau_n_tf) - (tau/(tau_n_tf-tau)) * tf.exp(-t_array/tau))
    )
    
    
    cost = error_TF(autocorr_array,autocorr_array_param)
    
    # Parameters for the optimizer
    f_tol = 1.0e-9
    g_tol = 1.0e-9
    max_iters = 500
    
    optimizer = ScipyOptimizerInterface(cost,method='L-BFGS-B',
                                    options={'disp': False,
                                            'maxls': 20,
                                            'iprint': 1,
                                            'gtol': g_tol,
                                            'eps': 1e-08,
                                            'maxiter': max_iters,
                                            'ftol': f_tol,
                                            'maxcor': 10,
                                            'maxfun': 15000})
    
    # Initialize all Tensors                
    init = tf.global_variables_initializer()
    
    # Optimization
    #with tf.Session(config=config) as session:
    with tf.Session() as session:        
        for i in range(n_init):
            session.run(init)
            
            params_init = np.squeeze(np.array([session.run(param_0),session.run(param_1),
                                    session.run(param_2),session.run(param_3)]))
            
            ## We run the optimizer
            optimizer.minimize(session)
            
            params_array[i] = np.squeeze(np.array([session.run(param_0),session.run(param_1),
                                    session.run(param_2),session.run(param_3)]))
            
            cost_array[i] = session.run(cost)
                
    gc.collect()
            
    best_iter = np.argmin(cost_array)
    cost = cost_array[best_iter]
    params = params_array[best_iter]
    
    alpha_opt = 1.0 / (1.0 + np.exp(params[0]))
    tau_opt = params[1] * params[1]
    tau_eta_opt = params[2] * params[2]
    f_osc_opt = 20.0 + params[3] * params[3]
    
    
    sd = np.std(cost_array)
    return (alpha_opt,tau_opt,tau_eta_opt,f_osc_opt,cost,sd)
    
    

############
#   Main   #
############

def main():
    
    in_location = "results/power_spectrum_along_PCS"
        
    out_location = in_location + "/oscillatoriness"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)  
    
    patterns = np.loadtxt(in_location+"/used_patterns").astype(int)
    print("Loading patterns: ", patterns)
    n_pat = len(patterns)
    
    n_pcs = 10
    
    n_init = 10 # Number of initial conditions for the fit
    
    select_PCs = np.arange(n_pcs)
    
    #eigenvalues = np.empty([n_pat,N_exc])
    #eigenvectors = np.empty([n_pat,N_exc,N_exc])
   
    oscillations_contribution = np.empty([n_pcs,8])
    
    oscillations_contribution_all = np.empty([n_pcs*n_pat,8])
    
    mean_oscillations_contribution = np.zeros((n_pcs,13))
    
    mean_oscillations_contribution[:,0] = np.arange(n_pcs)
    
    #for alpha in range(n_pat):
    #    eigenvalues[alpha] = np.loadtxt(in_location+"/eigenvalues_Sigma_"+str(patterns[alpha]))
    #    eigenvectors[alpha] = np.loadtxt(in_location+"/eigenvectors_Sigma_"+str(patterns[alpha]))
    
    done = 0
    
    #av_eival_dist = np.average(eigenvalues, axis =0)
    #np.savetxt(out_location+"/av_eival_dist", av_eival_dist)
        
    for pattern in range(n_pat):
        
        activity_from = patterns[pattern]
        
        PCs_from = patterns[pattern]
        
        tot_power = 0
            
        for pc in select_PCs:
            
            autocorr_file_name = ("autocorr_pat_"+ str(activity_from) 
                        + "_in_PC_"+str(pc) 
                        +"_of_pat_"+str(PCs_from))
            
            autocorr = np.loadtxt(in_location+"/" + autocorr_file_name)
            
            spectrum_file_name = ("spectrum_pat_"+ str(activity_from) 
                            + "_in_PC_"+str(pc) 
                            +"_of_pat_"+str(PCs_from))
                
            spectrum = np.loadtxt(in_location+"/" + spectrum_file_name)
            freqs = len(spectrum[:,0])
            total_power_pc = 0.5*np.sum(spectrum[:,1])/(freqs**2)
            tot_power += total_power_pc
            
            (alpha_opt,tau_opt,tau_eta_opt,f_osc_opt,cost,sd) = fit_autocorr_TF(autocorr, n_init)
            t_array = autocorr[:,0]
            
            autocorr_fit = np.copy(autocorr)
            autocorr_fit[:,1] = autocorr_param_numpy(t_array,alpha_opt,tau_opt,tau_eta_opt,f_osc_opt)
            
            oscillations_contribution[pc,0] = pc
            oscillations_contribution[pc,1] = total_power_pc
            oscillations_contribution[pc,2] = alpha_opt
            oscillations_contribution[pc,3] = tau_opt
            oscillations_contribution[pc,4] = tau_eta_opt
            oscillations_contribution[pc,5] = f_osc_opt
            oscillations_contribution[pc,6] = cost
            oscillations_contribution[pc,7] = sd
                        
            oscillations_contribution_all[done:done+1] = oscillations_contribution[pc]
            done += 1
            
            np.savetxt(out_location+"/"+autocorr_file_name+"_original",autocorr)
                
            np.savetxt(out_location+"/"+autocorr_file_name+"_best_fit",autocorr_fit)
            
            
        np.savetxt(out_location+"/oscillations_contribution_pat_"+ str(activity_from) 
                    + "_in_pcs_of_pat_"+str(PCs_from), oscillations_contribution)
        
        mean_oscillations_contribution[:,1] += oscillations_contribution[:,1]
        mean_oscillations_contribution[:,2] += oscillations_contribution[:,1]**2
        mean_oscillations_contribution[:,3] += oscillations_contribution[:,2]
        mean_oscillations_contribution[:,4] += oscillations_contribution[:,2]**2
        mean_oscillations_contribution[:,5] += oscillations_contribution[:,3]
        mean_oscillations_contribution[:,6] += oscillations_contribution[:,3]**2
        mean_oscillations_contribution[:,7] += oscillations_contribution[:,4]
        mean_oscillations_contribution[:,8] += oscillations_contribution[:,4]**2
        mean_oscillations_contribution[:,9] += oscillations_contribution[:,5]
        mean_oscillations_contribution[:,10] += oscillations_contribution[:,5]**2
        mean_oscillations_contribution[:,11] += (oscillations_contribution[:,1]/tot_power)
        mean_oscillations_contribution[:,12] += (oscillations_contribution[:,1]/tot_power)**2    
    
    
    mean_oscillations_contribution[:,1:] /= 1.0*n_pat
    mean_oscillations_contribution[:,2] = np.sqrt(mean_oscillations_contribution[:,2] - mean_oscillations_contribution[:,1]**2)
    mean_oscillations_contribution[:,4] = np.sqrt(mean_oscillations_contribution[:,4] - mean_oscillations_contribution[:,3]**2)
    mean_oscillations_contribution[:,6] = np.sqrt(mean_oscillations_contribution[:,6] - mean_oscillations_contribution[:,5]**2)
    mean_oscillations_contribution[:,8] = np.sqrt(mean_oscillations_contribution[:,8] - mean_oscillations_contribution[:,7]**2)
    mean_oscillations_contribution[:,10] = np.sqrt(mean_oscillations_contribution[:,10] - mean_oscillations_contribution[:,9]**2)
    mean_oscillations_contribution[:,12] = np.sqrt(mean_oscillations_contribution[:,12] - mean_oscillations_contribution[:,11]**2)
    
    np.savetxt(out_location+"/oscillations_contribution_all", oscillations_contribution_all)
    
    np.savetxt(out_location+"/mean_oscillations_contribution", mean_oscillations_contribution)
    
    
# Call Main
if __name__ == "__main__":
    main()
