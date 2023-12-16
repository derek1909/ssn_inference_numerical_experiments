'''
GSM with gabor filters

Created on Oct 6, 2016

@author: Rod
'''

import numpy as np
import scipy as sp
from scipy.stats import gamma, multivariate_normal
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, draw, show
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, os

import datetime

###################
#   Aux methods   #
###################

def circ_dist(i,j,m): # circular distance between points i and j (ring of size m)
    i_ = i%m
    j_ = j%m
    d = np.abs(i_-j_)
    if d > (m/2):
        return (m - d)-1
    else:
        return d

# Produces a matrix whose columns are a set of discreete fourier base vectors        
def get_fourier_base(m):
    base = np.empty([m,m])
    L = 2*np.pi/(1.0*m)
    
    for j in range(m):
        f = j//2
        if (j%2 == 0):
            for i in range(m):
                base[j][i] = np.cos(L*f*i)
        else:
            for i in range(m):
                base[j][i] = np.sin(L*f*i)
        norm = np.linalg.norm(base[j])
        if (norm != 0):
            base[j] = base[j]/norm
    return base.T

               
# Create an input covariance C with the right symmetry
# from the fourier base
def get_C_from_fourier(epsilon,base, decay_length):
    m = len(base)
    C = epsilon * np.identity(m) # Regularizer
    for i in range(m):
        v = base[:,i]
        lbda = 20*np.exp(-(1.0*(i//2))/decay_length)
        C = np.add(C,lbda*np.outer(v,v))
    C = C - 0.25    
    C =  C * (4.0/C[0,0])
    return C    
        
def test_matrices(A,C):
    eigenvalues, eigenvectors = np.linalg.eig(C)
    print("Eigenvalues of C:", eigenvalues)
        
    
    print("Rank of A = ", np.linalg.matrix_rank(A))

    np.savetxt(location+"/A", A)
    np.savetxt(location+"/C", C)
    
# If B contains the eigenvectors of M, then this method
# diagonalizes M
def diag_with_B(M,B):
    return np.dot(B.T,np.dot(M,B))

# Check orthonormality of B    
def check_orthonorm(B):
    m = len(B)
    N = np.empty([m,m])
    for i in range(m):
        for j in range(m):
            N[i][j] = np.dot(B[:,i],B[:,j])
    return N
    
###################
#   GSM methods   #
###################
    
# Get x from y and z

def get_x(y,z,A,s_x):
    x_mean = z*np.dot(A,y)
    noise = np.random.normal(scale=s_x, size=len(x_mean))
    return np.add(x_mean,noise)
    
# Functions to compute mean and covariance of the posterior

def get_post_moments(x,z_MAP,s_x_2,A,ATA,C_inv):
    Sigma = get_Sigma_z(z_MAP,s_x_2,C_inv,ATA)
    mu = get_mu_z(z_MAP,s_x_2,Sigma,A,x)
    return (mu, Sigma) 
    
def get_mu_z(z,s_x_2,Sigma_post,A,x):
    mu = (z/s_x_2)*np.dot(Sigma_post,np.dot(A.T,x))
    return mu 

def get_Sigma_z(z,s_x_2,C_inv,ATA):
    M = np.add(C_inv,(z*z/s_x_2)*ATA)
    return np.linalg.inv(M)
    
def get_post_moments_full_inference(x,p_z,s_x_2,A,ATA,C_inv):
    threshold = 10000.0
    dy = len(C_inv)
    n_points = len(p_z[0])
    mu_post = np.zeros(dy)
    Sigma_post = np.zeros((dy,dy))
    i_p_max = np.argmax(p_z[1])
    p_max = p_z[1,i_p_max]
    i_min = 0
    dz = p_z[0,1] - p_z[0,0]
    for i in range(i_p_max,-1,-1):
        if (p_z[1,i] <= (p_max/threshold)):
            i_min = i
            break
    i_max = n_points
    for i in range(i_p_max,n_points):
        if (p_z[1,i] <= (p_max/threshold)):
            i_max = i
            break
            
    for i in range(i_min,i_max):
        z = p_z[0,i]
        Sigma = get_Sigma_z(z,s_x_2,C_inv,ATA)
        mu = get_mu_z(z,s_x_2,Sigma,A,x)
        mu_post += (mu * p_z[1,i])
        # E[Var]_z + Var(mu)_z
        Sigma_post += ((Sigma + np.outer(mu,mu))* p_z[1,i])
    
    mu_post *= dz
    Sigma_post *= dz        
    # Substracting E(mu)E(mu)T
    Sigma_post -= np.outer(mu_post, mu_post)
    return (mu_post, Sigma_post) 
 
def get_post_moments_full_inference_logspace(x,p_z,s_x_2,A,ATA,C_inv): # This function assumes that z is logrithmically spaced
    threshold = 10000.0
    dy = len(C_inv)
    n_points = len(p_z[0])
    mu_post = np.zeros(dy)
    Sigma_post = np.zeros((dy,dy))
    i_p_max = np.argmax(p_z[1])
    p_max = p_z[1,i_p_max]
    i_min = 0
    n_contrasts = len(p_z[0])
    dchi = np.log10(p_z[0,1]) - np.log10(p_z[0,0])
    for i in range(i_p_max,-1,-1):
        if (p_z[1,i] <= (p_max/threshold)):
            i_min = i
            break
    i_max = n_points
    for i in range(i_p_max,n_points):
        if (p_z[1,i] <= (p_max/threshold)):
            i_max = i
            break
            
    for i in range(i_min,i_max):
        z = p_z[0,i]
        Sigma = get_Sigma_z(z,s_x_2,C_inv,ATA)
        mu = get_mu_z(z,s_x_2,Sigma,A,x)
        mu_post += (mu * p_z[1,i] * z)
        # E[Var]_z + Var(mu)_z
        Sigma_post += ((Sigma + np.outer(mu,mu))* p_z[1,i] * z)
        
    mu_post *= np.log(10.0) * dchi
    Sigma_post *= np.log(10.0) * dchi
    Sigma_post -= np.outer(mu_post, mu_post)
    return (mu_post, Sigma_post) 
 

# Get the eigenvalues of Sigma as a function of those 
# of C and ATA    
def get_eigen_post_Sigma(z,s_x_2,eigen_C,eigen_ATA):
    s = (z*z)/s_x_2
    eigen_inv = np.add(np.reciprocal(eigen_C), s* eigen_ATA)
    return np.reciprocal(eigen_inv)

def reset_sigma_x (eig_C, eig_ATA):
    s_x_2 = eig_C * eig_ATA / (2*eig_C -1.0)
    s_x = np.sqrt(s_x_2)
    return (s_x, s_x_2)
    
def reset_sigma_x_w_deriv (eig_C, eig_ATA, z_max):
    d_eig_C = eig_C[2] - eig_C[0]
    d_eig_ATA = eig_ATA[2] - eig_ATA[0]
    
    s_x_2 = (eig_C[0]/z_max)**2.0  * (d_eig_ATA / d_eig_C)
    s_x = np.sqrt(s_x_2)
    return (s_x, s_x_2)
    
def P_z (z, k, theta):
    return gamma.pdf(z, k, loc=0, scale=theta)

def log_P_z (z, k, theta):
    return gamma.logpdf(z, k, loc=0, scale=theta)

def P_z_giv_x (z_range,x,ACAT,s_x_2, k, theta):
    n_contrasts = len(z_range)
    D_x = len(x)
    log_p = np.empty([n_contrasts])
    p = np.empty([2,n_contrasts])
    mean = np.zeros([D_x])
    dz = z_range[1] - z_range[0]
    for i in range(n_contrasts):
        Cov = np.add(z_range[i]*z_range[i]*ACAT,s_x_2*np.identity(D_x))
        log_p[i] = log_P_z(z_range[i], k, theta) + multivariate_normal.logpdf(x, mean, Cov)
    
    max_lp = np.amax(log_p)
    p[0] = z_range
    p[1] = np.exp(log_p-max_lp)
    norm = np.sum(p[1]) * dz
    p[1] = p[1]/norm
    
    return p   
    
############
#   Main   #
############

def main():
    load_images = False # If true we load the images (x_array) from image_location
    
    baseline = 3.0 # We add this to the means only in the targets folder
    
    gamma = 0.0 #0.75 # contrast-proportional contribution to h
    
    input_case = "bumps"
    
    np.random.seed(seed=912345678)
    
    x_noise = False # If true we add noise to x
    
    if x_noise:
        subfolder = input_case+"/w_noise/"
    else:
        subfolder = input_case+"/no_noise/"
        
    results_location = subfolder+"results"
    target_location_full_inf = subfolder+"targets_full_inf"
    target_location_map_inf = subfolder+"targets_map_inf"    
    
    if not os.path.exists(results_location):
        os.makedirs(results_location)
        
    
    if not os.path.exists(target_location_full_inf):
        os.makedirs(target_location_full_inf)
    
    if not os.path.exists(target_location_map_inf):
        os.makedirs(target_location_map_inf)
    
    FILTER_FILE = "filters.npy"
    
    text_file = open(results_location+"/000_filter_used.txt", "w")
    text_file.write("Filter file used: %s" % FILTER_FILE)
    text_file.close()
    
    print("Start time : ", datetime.datetime.now())
        
    # We first import the filters, since this determines the dimensionality

    A = np.load(FILTER_FILE)
    ATA = np.dot(A.T,A)
    
    
    D_x = len(A)              # Dimensionality of the observed variable x
    D_y = len(A[0])           # Dimensionality of the hidden variable y
    L = int(np.sqrt(D_x))          # Side length of the square image represented by x
    
    h_scale = 1.0/15.0
    
    # We create the basis matrix B
    B = get_fourier_base(D_y)
    
    # We build C from B
    decay_length = D_y / 50.0 # for the eigenvalues of C
    epsilon = 0.01 # regularizer for C
    C = get_C_from_fourier(epsilon,B, decay_length)
    
    d = 0.0 * np.sqrt(C[0,0]) # average mean of y in units of sd's 
    
    ACAT = np.dot(np.dot(A,C),A.T)
    
    # We generate the dataset
    
    n_targets = 6            # N° of points in the dataset (or total observation time)
    
    # Dist params:
    s_x = 10.0                # Noise of the x process
    s_x_2 = s_x**2
    k = 2.0                   # Shape parameter of the gamma dist. for z
    theta = 2.0               # Scale parameter of the gamma dist. for z
    
    dist_params = np.array([s_x,k,theta]) # I pack them for convenient saving and loading
    
    
    # Mean and covariance of y:
    y_mean = np.zeros(D_y)            # The mean is 0 for y

    # We will need the inverse of C
    C_inv = np.linalg.inv(C)
  
    
    x_array = np.empty([n_targets,D_x])
    
    if load_images:
        for alpha in range(n_targets):
            x_array[alpha] = np.loadtxt(results_location+"/x_"+str(alpha))
            
    else:
        
        # Drawing samples for y and setting contrast levels (we fix those)
        y_array = np.empty([n_targets,D_y])
        #z_array = np.random.gamma(shape = k, scale=theta, size = n_targets)
        
        if input_case == "sampled": z_array = np.array([0.0,0.0,0.5,0.5,1.0,1.0])
        else: z_array = np.array([0.0,0.125,0.25,0.5,1.0,2.0])
        print(z_array)
        np.savetxt(results_location+"/z_array",z_array)
        
        if input_case == "sampled":
            print("Sampling y")
            for alpha in range(n_targets):
                y_array[alpha] = np.random.multivariate_normal(mean = y_mean + d, cov = C, size = 1) 
                      
        elif input_case == "bumps":
            center = D_y//2
            sigma = 0.15*D_y
            print("Creating y bumps")
            for i in range(D_y):
                y_array[0,i] = 6*np.exp(-0.5*((i-center)/sigma)**2.0)
            
            y_array[0] -= np.mean(y_array[0])
            y_array[0] *= (np.sqrt(C[0,0])/np.std(y_array[0]))
            y_array[0] += d
            for alpha in range(n_targets):
                y_array[alpha] = y_array[0]
                
        else:
            print("Unknown input case!")
            sys.exit()
        
        # We now get the x samples
        if x_noise:
            print("Obtaining x from y (with noise)")
            for alpha in range(n_targets):
                x_array[alpha] = get_x(y_array[alpha],z_array[alpha],A,s_x)
        else:
            print("Obtaining x from y (without noise)")
            for alpha in range(n_targets):
                x_array[alpha] = z_array[alpha]*np.dot(A,y_array[alpha])
    
        
        scale = np.amax(np.abs(x_array))
        
        for alpha in range(n_targets):
            np.savetxt(results_location+"/x_"+str(alpha),x_array[alpha])
            np.savetxt(results_location+"/y_"+str(alpha),y_array[alpha])
           
    # We compute the moments of the posteriors
    
    mu_post_array_true_z = np.empty([n_targets,D_y])
    Sigma_post_array_true_z = np.empty([n_targets,D_y,D_y])
    std_post_array_true_z = np.empty([n_targets,D_y])
    
    mu_post_array_z_MAP = np.empty([n_targets,D_y])
    Sigma_post_array_z_MAP = np.empty([n_targets,D_y,D_y])
    std_post_array_z_MAP = np.empty([n_targets,D_y])
    
    mu_post_array_full_inf = np.empty([n_targets,D_y])
    Sigma_post_array_full_inf = np.empty([n_targets,D_y,D_y])
    std_post_array_full_inf = np.empty([n_targets,D_y])
    
    
    h_array = np.empty([n_targets,D_y])
    h_twice_array = np.empty([n_targets,2*D_y])
    
    z_min = 0.0
    z_max = 5.0
    n_points = 201
    z_range = np.linspace(z_min, z_max, num=n_points, endpoint=True)
        
    h_true = np.empty([n_targets,2*D_y])
    
    results = np.empty([n_targets,11])
    
    z_MAP_array = np.empty([n_targets])
    
    print("Computing posterior moments")
    
    for alpha in range(n_targets):
        print("Target "+str(alpha))
        p_z = P_z_giv_x(z_range,x_array[alpha],ACAT,s_x_2, k, theta)
        np.savetxt(results_location+"/p_z_giv_x_"+str(alpha),p_z.T)
        
        # Moments assuming we know the true contrast
        mu_post_array_true_z[alpha],Sigma_post_array_true_z[alpha] = get_post_moments(
                                                    x_array[alpha],z_array[alpha],s_x_2,A,ATA,C_inv)
        std_post_array_true_z[alpha] = np.sqrt(np.diag(Sigma_post_array_true_z[alpha]))
        
        # Moments at MAP contrast
        z_MAP_array[alpha] = p_z[0][np.argmax(p_z[1])]
        
        mu_post_array_z_MAP[alpha],Sigma_post_array_z_MAP[alpha] = get_post_moments(
                                                x_array[alpha],z_MAP_array[alpha],s_x_2,A,ATA,C_inv)
        std_post_array_z_MAP[alpha] = np.sqrt(np.diag(Sigma_post_array_z_MAP[alpha]))
        
        # Full inference 
        mu_post_array_full_inf[alpha],Sigma_post_array_full_inf[alpha] = get_post_moments_full_inference(x_array[alpha],p_z,s_x_2,A,ATA,C_inv)
        
        std_post_array_full_inf[alpha] = np.sqrt(np.diag(Sigma_post_array_full_inf[alpha]))
        
        
        h_array[alpha] =  h_scale*(np.dot(A.T,x_array[alpha])+ gamma * np.linalg.norm(x_array[alpha]))
        h_twice_array[alpha][0:D_y] = h_array[alpha]
        h_twice_array[alpha][D_y:2*D_y] = h_twice_array[alpha][0:D_y]
                
        results[alpha][0] = z_array[alpha]
        results[alpha][1] = z_MAP_array[alpha]
        results[alpha][2] = np.average(mu_post_array_true_z[alpha])
        results[alpha][3] = np.amax(mu_post_array_true_z[alpha])
        results[alpha][4] = np.average(std_post_array_true_z[alpha])
        results[alpha][5] = np.average(mu_post_array_z_MAP[alpha])
        results[alpha][6] = np.amax(mu_post_array_z_MAP[alpha])
        results[alpha][7] = np.average(std_post_array_z_MAP[alpha])
        results[alpha][8] = np.average(mu_post_array_full_inf[alpha])
        results[alpha][9] = np.amax(mu_post_array_full_inf[alpha])
        results[alpha][10] = np.average(std_post_array_full_inf[alpha])
    
               
    # Save everything 
    np.savetxt(results_location+"/C",C)
    np.savetxt(results_location+"/A",A)
    np.savetxt(results_location+"/dist_params",dist_params)
    np.savetxt(results_location+"/mu_and_std_vs_contrast_gsm",results)
    np.savetxt(results_location+"/z_MAP_array",z_MAP_array)
    
    for alpha in range(n_targets):
        np.savetxt(results_location+"/h_"+str(alpha),h_array[alpha])
        np.savetxt(results_location+"/h_twice_"+str(alpha),h_twice_array[alpha])
        
        np.savetxt(results_location+"/mu_true_z_"+str(alpha),mu_post_array_true_z[alpha])
        np.savetxt(results_location+"/Sigma_true_z_"+str(alpha),Sigma_post_array_true_z[alpha])
        np.savetxt(results_location+"/std_true_z_"+str(alpha),std_post_array_true_z[alpha])
        
        np.savetxt(results_location+"/mu_z_MAP_"+str(alpha),mu_post_array_z_MAP[alpha])
        np.savetxt(results_location+"/Sigma_z_MAP_"+str(alpha),Sigma_post_array_z_MAP[alpha])
        np.savetxt(results_location+"/std_z_MAP_"+str(alpha),std_post_array_z_MAP[alpha])
        
        np.savetxt(results_location+"/mu_full_inf_"+str(alpha),mu_post_array_full_inf[alpha])
        np.savetxt(results_location+"/Sigma_full_inf_"+str(alpha),Sigma_post_array_full_inf[alpha])
        np.savetxt(results_location+"/std_full_inf_"+str(alpha),std_post_array_full_inf[alpha])
        
        np.savetxt(target_location_full_inf+"/h"+str(alpha),h_twice_array[alpha])
        np.savetxt(target_location_full_inf+"/mu"+str(alpha),(baseline + mu_post_array_full_inf[alpha]))
        np.savetxt(target_location_full_inf+"/sigma"+str(alpha),Sigma_post_array_full_inf[alpha])
        np.savetxt(target_location_full_inf+"/std"+str(alpha),std_post_array_full_inf[alpha])
        
        np.savetxt(target_location_map_inf+"/h"+str(alpha),h_twice_array[alpha])
        np.savetxt(target_location_map_inf+"/mu"+str(alpha),(baseline + mu_post_array_z_MAP[alpha]))
        np.savetxt(target_location_map_inf+"/sigma"+str(alpha),Sigma_post_array_z_MAP[alpha])
        np.savetxt(target_location_map_inf+"/std"+str(alpha),std_post_array_z_MAP[alpha])
        
   
    print("End time : ", datetime.datetime.now())
    
# Call Main
if __name__ == "__main__":
    main()  
