import numpy as np
import scipy as sp
import methods as mt
from parameters import *
import datetime, os
    
    
# Definition of autocorrelation
    
def autocorr(x):
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean)/std
    result = np.correlate(x, x, "same")/len(x)
    return (result[result.size//2:])

############
#   Main   #
############

def main():
    
    u_threshold = -65.0e-3
    V_rev_E = 0.0
    V_rev_I = -80.0e-3
    
    C_membrane = 20.0e-12
    
    from_trainset = False

    if (from_trainset):
        final_contrast_level = 3
    else:
        final_contrast_level = 10
        contrast = 0.35
#        final_contrast_level = 5
#        contrast = 2.0
    
    net_params_location = "parameter_files"
    in_location = "results/transient"
    
    out_location = in_location
            
    if not os.path.exists(out_location):
        os.makedirs(out_location) 
    
    W = np.loadtxt(net_params_location+"/w_learn")
    
    stats = 20
            
    t_init = 225.0e-3 
    t_stimulus = 100.0e-3 
    t_final = 125.0e-3
    
    total_time = t_init + t_stimulus + t_final
    t_bet_samp = 2*dt
    steps_bet_samp = int(t_bet_samp/dt)    
      
    
    points_init = int(t_init/t_bet_samp)
    points_stimulus = int(t_stimulus/t_bet_samp)
    points_final = int(t_final/t_bet_samp)
    
    sample_size = points_init + points_stimulus + points_final
    
    samp_rec = N
    
    T_window = 10.0 * tau_e # 200ms
    points_per_window = int(T_window/t_bet_samp)
    
    keep_time = total_time#1.5*T_window
    keep_points = sample_size#int(1.5 * points_per_window)
    
    
    
    ts = t_bet_samp
    
    inputs = np.empty([stats,keep_points,2*N+1])
    conductances = np.empty([stats,keep_points,2*N+1])
    
    for t in range(keep_points): inputs[:,t,0] = t * ts
    conductances[:,:,0] = inputs[:,:,0]   
    
    print("Start time : ", datetime.datetime.now())

    h_init = np.loadtxt(net_params_location+"/h_true_"+str(0)+"_learn")
	
    if (from_trainset):
        h_final = np.loadtxt(net_params_location+"/h_true_"+str(final_contrast_level)+"_learn")
        
    else:
        input_scaling_learn = np.loadtxt(net_params_location+"/input_scaling_learn")
        input_baseline_learn = np.loadtxt(net_params_location+"/input_baseline_learn")
        input_nl_pow_learn = np.loadtxt(net_params_location+"/input_nl_pow_learn")
        h_full_contrast = np.loadtxt(net_params_location+"/h4")
        h_lin= contrast * h_full_contrast
        h_final = input_scaling_learn * np.power((h_lin + input_baseline_learn),input_nl_pow_learn)
        
    u_samples = np.load(in_location+"/u_samples_transient_0_"+str(final_contrast_level)+".npy")
    r_samples = np.load(in_location+"/r_samples_transient_0_"+str(final_contrast_level)+".npy")
    eta_samples = np.load(in_location+"/eta_samples_transient_0_"+str(final_contrast_level)+".npy")
    
    delays = np.loadtxt(in_location+"/delays_transient_0_"+str(final_contrast_level))
    
    r_samples = r_samples[:stats,:keep_points]
    u_samples = u_samples[:stats,:keep_points]
    eta_samples = eta_samples[:stats,:keep_points]
    u_samples = u_threshold + 1.0e-3 * u_samples
    
    
    
    #r_samples = np.average(r_samples[:,:keep_points,:], axis =0)
    #eta_samples = np.average(eta_samples[:,:keep_points,:], axis =0)
    h = np.empty([keep_points,N])
    h[:points_init+points_stimulus] = h_init
    
    for i in range(points_stimulus):
        time = i *t_bet_samp
        arrived = np.where(delays <= time)
        h[i+points_init,arrived] = h_final[arrived]
    
    h[points_init+points_stimulus:] = h_final
    
    for i in range(points_final):
        time = i *t_bet_samp
        arrived = np.where(delays <= time)
        h[i+points_init+points_stimulus,arrived] = h_init[arrived]
    
    r_samples_e = r_samples[:,:,:N_exc]
    r_samples_i = r_samples[:,:,N_exc:]
          
    
    W_e_T = (W[:,:N_exc]).T
    W_i_T = (W[:,N_exc:]).T
    print("Computing conductance history...")
    
    for s in range(stats):
    
        external_input = h + eta_samples[s]
    
        positive_input_indexes = external_input >= 0.0
        negative_input_indexes = external_input < 0.0
        
        # External
        input_e_ext = np.zeros([keep_points,N])
        input_i_ext = np.zeros([keep_points,N])
    
        input_e_ext[positive_input_indexes] = external_input[positive_input_indexes]
        input_i_ext[negative_input_indexes] = external_input[negative_input_indexes]
        
        # Recurrent
        input_e_rec = r_samples_e[s] @ W_e_T # Size = samples x N
        input_i_rec = r_samples_i[s] @ W_i_T # Size = samples x N
    
        # All
        input_e = input_e_rec + input_e_ext
        input_i = input_i_rec + input_i_ext
        
        inputs[s,:,1:N+1] = input_e
        inputs[s,:,N+1:] = input_i
        
        g_e = C_membrane * tau_inv.T * np.divide(input_e, V_rev_E - u_samples[s])
        g_i = C_membrane * tau_inv.T * np.divide(input_i, V_rev_I - u_samples[s])
    
        conductances[s,:,1:N+1] = g_e
        conductances[s,:,N+1:] = g_i
    
    
    inputs_av = np.average(inputs, axis =0)
    inputs_std = np.std(inputs, axis =0)
    inputs_sem = inputs_std / np.sqrt(stats)
    conductances_av = np.average(conductances, axis =0)
    conductances_std = np.std(conductances, axis =0)
    conductances_sem = conductances_std / np.sqrt(stats)         
    print("End time : ", datetime.datetime.now())
    
    np.savetxt(out_location+"/e_i_inputs_transient_0_"+str(final_contrast_level)+"_w_input", np.concatenate((inputs_av,inputs_std[:,1:],inputs_sem[:,1:]), axis=1))
        
    np.savetxt(out_location+"/e_i_conductances_transient_0_"+str(final_contrast_level)+"_w_input", np.concatenate((conductances_av,conductances_std[:,1:],conductances_sem[:,1:]), axis=1))
    
    conductances_change_av = np.copy(conductances_av)
    
    conductances_change_av[:,1:] -= np.average(conductances_av[:points_init,1:],axis =0)
    
    np.savetxt(out_location+"/e_i_conductances_change_transient_0_"+str(final_contrast_level)+"_w_input", np.concatenate((conductances_change_av,conductances_std[:,1:],conductances_sem[:,1:]), axis=1))
    
    
    
    

# Call Main
if __name__ == "__main__":
    main()
