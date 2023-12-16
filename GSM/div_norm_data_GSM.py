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
from plotting_methods import mat_plot, multimat_plot
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
        
    mat_plot(C,"C",np.amax(np.abs(C)),location)
    mat_plot(A,"A",np.amax(np.abs(A)),location)

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

def np_th(u):
    return np.maximum(u,0)
    

def get_true_h_from_x_proj(x_proj,h_scale,input_scaling,input_baseline,input_pow):
    # h = h_scale * x_proj
    # true_h = input_scaling*(input_baseline + h)**input_pow
    h = h_scale*x_proj
    true_h = input_scaling*((input_baseline + h)**input_pow)
    return np.concatenate([true_h,true_h])
    
############
#   Main   #
############

def main():
    load_images = False # If true we load the images (x_array) from image_location
    
    baseline = 3.0 # We add this to the means only in the targets folder
    
    np.random.seed(seed=912345678)
    
    x_noise = False # If true we add noise to x
    
    N_points_per_contrast = 200
    
    if x_noise:
        subfolder = "div_norm_data/w_noise/"
    else:
        subfolder = "div_norm_data/no_noise/"
        
    results_location = subfolder+"results_GSM"
    targets_location = subfolder+"targets_GSM"
    
    in_location = "../SSN/ssn_evolution/results_net"
    #in_location = "COSYNE/results"
    
    if not os.path.exists(results_location):
        os.makedirs(results_location)
    
    if not os.path.exists(targets_location):
        os.makedirs(targets_location)    
    
    print("Start time : ", datetime.datetime.now())
        
    # We first import the filters, since this determines the dimensionality

    A = np.loadtxt(in_location+"/A")
    ATA = np.dot(A.T,A)
    
    D_x = len(A)              # Dimensionality of the observed variable x
    D_y = len(A[0])           # Dimensionality of the hidden variable y
    L = int(np.sqrt(D_x))          # Side length of the square image represented by x
    
    h_scale = 1.0/15.0
    
    input_scaling = np.loadtxt(in_location+"/input_scaling_learn")
    input_baseline = np.loadtxt(in_location+"/input_baseline_learn")
    input_pow = np.loadtxt(in_location+"/input_nl_pow_learn")
    
    C = np.loadtxt(in_location+"/C")
    
    ACAT = np.dot(np.dot(A,C),A.T)
    
    # We generate the dataset
    
    n_targets = 5            # N° of points in the dataset (or total observation time)
    
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
    
    z_array = np.array([0.0,0.125,0.25,0.5,1.0])
    
    original_bump = np.loadtxt(in_location+"/y_0")
    mean_array = np.full((D_y),np.average(original_bump))
    
    mu_post_array_true_z = np.empty([n_targets,D_y])
    Sigma_post_array_true_z = np.empty([n_targets,D_y,D_y])
    std_post_array_true_z = np.empty([n_targets,D_y])
    
    mu_post_array_z_MAP = np.empty([n_targets,D_y])
    Sigma_post_array_z_MAP = np.empty([n_targets,D_y,D_y])
    std_post_array_z_MAP = np.empty([n_targets,D_y])
    
    mu_post_array_full_inf = np.empty([n_targets,D_y])
    Sigma_post_array_full_inf = np.empty([n_targets,D_y,D_y])
    std_post_array_full_inf = np.empty([n_targets,D_y])
    
    h_array = np.empty([n_targets,2*D_y])
    
    z_min = 0.0
    z_max = 5.0
    n_points = 201
    z_range = np.linspace(z_min, z_max, num=n_points, endpoint=True)
        
    h_true = np.empty([n_targets,2*D_y])
    
    z_MAP_array = np.empty([n_targets])
    
    for s in range (N_points_per_contrast): 
        y_array = np.random.multivariate_normal(mean = mean_array, cov = C, size = n_targets) 
        
        for alpha in range(n_targets):
            point_number = alpha + s * n_targets
            print("point "+str(point_number))
            
            x_array[alpha] = z_array[alpha]*np.dot(A,y_array[alpha]) # no readout noise
            h_array[alpha] = get_true_h_from_x_proj(np.dot(A.T,x_array[alpha]),h_scale,
                                        input_scaling,input_baseline,input_pow)
            
            #while(any(np.isnan(h_array[alpha]))):
            #    print("Redoing one. Contained nans.")
            #    y_array[alpha] = np.random.multivariate_normal(mean = mean_array, cov = C, size = 1)      
            #    x_array[alpha] = z_array[alpha]*np.dot(A,y_array[alpha]) # no readout noise
            #    h_array[alpha] = get_true_h_from_x_proj(np.dot(A.T,x_array[alpha]),h_scale,
            #                            input_scaling,input_baseline,input_pow)
            
            if (any(np.isnan(h_array[alpha]))): print("NAAAAAN!!!!")
                
            np.savetxt(results_location+"/x_"+str(point_number),x_array[alpha])
            np.savetxt(results_location+"/y_"+str(point_number),y_array[alpha])
            
            p_z = P_z_giv_x(z_range,x_array[alpha],ACAT,s_x_2, k, theta)
            
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
            
            np.savetxt(results_location+"/x_"+str(point_number),x_array[alpha])
            np.savetxt(results_location+"/y_"+str(point_number),y_array[alpha])
            np.savetxt(results_location+"/h_true_"+str(point_number),h_array[alpha])
            
            np.savetxt(results_location+"/mu_true_z_"+str(point_number),mu_post_array_true_z[alpha])
            np.savetxt(results_location+"/Sigma_true_z_"+str(point_number),Sigma_post_array_true_z[alpha])
            np.savetxt(results_location+"/std_true_z_"+str(point_number),std_post_array_true_z[alpha])
            
            np.savetxt(results_location+"/mu_z_MAP_"+str(point_number),mu_post_array_z_MAP[alpha])
            np.savetxt(results_location+"/Sigma_z_MAP_"+str(point_number),Sigma_post_array_z_MAP[alpha])
            np.savetxt(results_location+"/std_z_MAP_"+str(point_number),std_post_array_z_MAP[alpha])
            
            np.savetxt(results_location+"/mu_full_inf_"+str(point_number),mu_post_array_full_inf[alpha])
            np.savetxt(results_location+"/Sigma_full_inf_"+str(point_number),Sigma_post_array_full_inf[alpha])
            np.savetxt(results_location+"/std_full_inf_"+str(point_number),std_post_array_full_inf[alpha])
            
            
            np.savetxt(targets_location+"/x_"+str(point_number),x_array[alpha])
            np.savetxt(targets_location+"/y_"+str(point_number),y_array[alpha])
            np.savetxt(targets_location+"/h"+str(point_number),h_array[alpha])        
            np.savetxt(targets_location+"/mu"+str(point_number),baseline + mu_post_array_true_z[alpha])
            np.savetxt(targets_location+"/sigma"+str(point_number),Sigma_post_array_true_z[alpha])
            np.savetxt(targets_location+"/std"+str(point_number),std_post_array_true_z[alpha])
    
    
    
    
    print("End time : ", datetime.datetime.now())
    
# Call Main
if __name__ == "__main__":
    main()  
