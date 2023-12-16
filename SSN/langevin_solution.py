# Find matrices for equivalent Langevin Sampler to the SSN solution

import numpy as np
import scipy as sp
from scipy.linalg import sqrtm , solve_lyapunov
import methods as mt
from parameters import *

import datetime, os

############
#   Main   #
############

def main():
    in_location = "parameter_files"
    out_location = "results/langevin"
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
     
    A_ext = np.zeros([2*N,2*N])
    B_ext = np.zeros([2*N,2*N])
    
    A_ext[:N,N:] = id_N
    A_ext[N:,N:] = -tau_n_inv*id_N
    
    Sigma_eta = np.loadtxt(in_location+"/sigma_eta_learn")
    #np.savetxt(out_location+"/Sigma_eta", Sigma_eta)
    
    B = tau_n*Sigma_eta
    B_ext[N:,N:] = B
    
    np.savetxt(out_location+"/B", B)
    
    for alpha in range(N_pat):
        Sigma = np.loadtxt(in_location+"/sigma_evolved_net_"+str(alpha))
        Sigma_inv = np.linalg.inv(Sigma)
        A = 0.5 * tau_lang*tau_n_inv*(id_N - sqrtm(id_N+ ((2.0*tau_n*tau_lang_inv)**2.0) * (Sigma_eta @ Sigma_inv)))
        
        eigenvalues,_ = np.linalg.eig(A)
        ei_max = np.amax(eigenvalues.real)
        print("spectral abscissa of A = ", ei_max)
        
        np.savetxt(out_location+"/A_"+str(alpha),A)
        A_ext[:N,:N] = A
        #np.savetxt(out_location+"/A_ext_"+str(alpha),A_ext)
        Sigma_solve = solve_lyapunov(A_ext, -2.0*tau_n_inv*tau_n_inv*B_ext)
        #np.savetxt(out_location+"/Sigma_"+str(alpha),Sigma)
        np.savetxt(out_location+"/Sigma_solve_"+str(alpha),Sigma_solve)
        print(np.linalg.norm(Sigma - Sigma_solve[:N,:N]))
        A_Sigma = A_ext @ Sigma_solve
        L = A_Sigma + A_Sigma.T + 2.0*tau_n_inv*tau_n_inv*B_ext
        print(np.linalg.norm(L))
# Call Main
if __name__ == "__main__":
    main()
