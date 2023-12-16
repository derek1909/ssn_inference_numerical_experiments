import numpy as np
import methods as mt
import os

############
#   Main   #
############

def main():

    oscillations = True
    strong = True
    
    if (oscillations):
        f = 40 # in Hz
        if strong:
            out_location ="samples_strong_osc"    
        else:
            out_location = "samples_osc"
    else:
        out_location = "samples_no_osc"
        f = 0 # in Hz
        strong = False
    
    compute_autocorr_and_spectrum = True
    
    if not os.path.exists(out_location):
        os.makedirs(out_location) 
    
    epsilon = 10.e-10
    
    s_n2 = 0.0  # sqr noise amplitude
    s2 = 4.0 - s_n2
    tau = 20.0e-3 # in s
    
    n = 2000
    spacing = 0.5e-3 # in s
    T = (n-1) * spacing
    
    n_draws = 100000 # number of draws from the distribution
        
    X = np.expand_dims(np.linspace(start = 0.0, stop = T, num=n, endpoint=True), axis = 0)   
    
    mean = np.zeros([n])
    print("computing covariance matrix...")
    cov = mt.compute_cov(s2, tau, s_n2, epsilon, X, strong, f)
    
    print("drawing samples")
    samples = np.random.multivariate_normal(mean, cov, n_draws)
    
    np.savetxt(out_location+"/covariance", cov) 
    np.savetxt(out_location+"/samples", np.append(X.T,samples.T,axis=1))        
        
    if (compute_autocorr_and_spectrum == True):
        print("computing autocorrelation and power spectrum")
        times = n//2 
        freqs = n//2 + 1
        
        spectrum = np.zeros([freqs,2])
        autocorr = np.zeros([times,2])
        
        ts = spacing
        fs = 1.0/T
        
        for i in range(times): autocorr[i,0] = ts * i
            
        for i in range(freqs): spectrum[i,0] = fs * i        
        
        mean_samps = np.mean(samples)
        std_samps = np.std(samples)
        
        for samp in range(n_draws):
            autocorr[:,1] += mt.autocorr(samples[samp],mean_samps,std_samps)
            spectrum[:,1] += np.absolute(np.fft.rfft(samples[samp]))**2.0
        
        spectrum[:,1] = spectrum[:,1] / (1.0*n_draws)
        autocorr[:,1] = autocorr[:,1] / (1.0*n_draws)
        
    np.savetxt(out_location+"/autocorr", autocorr)
    np.savetxt(out_location+"/spectrum", spectrum)               
# Call Main
if __name__ == "__main__":
    main()  
