import numpy as np

##################
#   Parameters   #
##################


# Patterns
N_pat = 5                       # N° of input patterns

# Input noise parameters

white = False                    # If True, the input noise is temporally white
#tau_n = 50.0e-3                     # Timescale of the noise correlation (if white = False)
tau_n = 20.0e-3  
if not (tau_n == 20.0e-3):
    print("WARNING: tau_eta = ", tau_n)

# Network parameters
N = 100                          # N° of neurons in the net (Better to use a multiple of 5)
N_cons = 50                      # Dimension of the constrained part of the output
r_E_I = 1.0                     # Ratio of exc/inh neurons 
f_E =  r_E_I / (1.0 + r_E_I)    # Fraction of excitatory neurons
f_I = 1.0 / (1.0 + r_E_I)       # Fraction of inhibitory neurons
N_exc = int(f_E * N)            # N° of excitatory neurons
N_inh = int(f_I * N)            # N° of inhibitory neurons

Sign = np.full((1,N), 1.0)           # We asign signs to synapes
for i in range(N_inh):
    Sign[0][N-1-i] = -int(r_E_I)    # The factor is to keep the net balanced
Sign_inv = np.reciprocal(Sign)

weight_scale = 0.01/N            # Scale for initially drawing random weights

chol_length = int((N*(N+1))/2.0)   # Number of non-zero elements in the cholesky factors of Sigma
chol_tgt_length = int((N_cons*(N_cons+1))/2.0)   # Number of non-zero elements in the cholesky factors of Sigma tgt
chol_aux_length = chol_length - chol_tgt_length # Number of auxiliary elements in the cholesky factors

# Neural parameters
n = 2                          # n, k of the nonlinearity (n = -1, gives you the linear case)
k = 0.3

tau_e = 20.0e-3
tau_i = 10.0e-3

tau = np.full((N,1), 1.0)  # Array containing the taus
for i in range(N):
    if (i < N_exc):
        tau[i][0] = tau_e
    else:
        tau[i][0] = tau_i

tau_inv = np.reciprocal(tau)    # Inverse taus
tau_n_inv = 1.0 /(1.0* tau_n)

dt = 0.2e-3                      # Integration time-step for the evolution of the moments and network
#dt = 0.05e-3 

if ((dt < 0.19e-3) or (dt > 0.21e-3)):
    print("WARNING: dt is not 0.2ms")

eps1 = 1. - dt / tau_n
eps2 = dt * (1. + dt / tau_n)

iter_exp = 10                   # Number of iterations to approximate the exp

alpha_slc = 0.01                     # Relative weight of the slowing cost

N_inv = 1.0/(1.0*N)
N_2_inv = 1.0/(1.0*N*N)

# Parameters for the inv. Wishart Dist.

deg_free = N_cons * 5

id_N = np.identity(N)
tau_n_inv_id_N = tau_n_inv * id_N

zero_Mat = 0.0 * id_N

# Parameters for time-modulated transient

tau_t = 10.0e-3
cuad_factor = 11.0 #4.0

# Parameters for the langevin sampler

tau_lang = tau_e
tau_lang_inv = 1.0/tau_lang

zero_vec = np.zeros(N)

