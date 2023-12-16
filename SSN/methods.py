from scipy.special import erf
from scipy.linalg import expm as matexp
from scipy.stats import wishart
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from parameters import *

#############################
#   Functions and Methods   #
#############################

# Standard normal distribution
def f1(x): 
    return np.exp(-x*x/2.0)/np.sqrt(2*np.pi)

def f2_exact(x): # Cumulative distribution function for the standard normal distribution
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0

# Squared Frobenius norm
def sqr_norm(x):
    return np.sum(x*x)
    
# Rescale an array to have a desired norm    
def rescale(array, norm):
    return array * norm / np.linalg.norm(array)

def rel_dist(array,array_tgt):
    return np.linalg.norm(array-array_tgt) / np.linalg.norm(array_tgt)

#-------------------------------#
# Modulation overshoot in input #
#-------------------------------#

def h_modulation(t):
    return ((1.0 - np.exp(-t/tau_t)) / (1.0 + np.exp(-t/tau_t)
                -np.exp(-t**2.0/(cuad_factor*tau_t**2.0))))
    
#-------------------------#
# Rates and their moments #
#-------------------------#

# Return output rate for mem. pot. u
def get_r(u):
    if n == -1:
        return u
    else:
        return k*np.power(0.5*(u+np.absolute(u)),n)
        

# Return derivative of the output rate wrt u
def get_r_prime(u):
    if n == -1:
        return np.full((N), 1.0)
    else:
        return n*k*np.power(0.5*(u+np.absolute(u)),(n-1))
        
# Return mean rates nu
def get_nu(mu, diag_Sigma): 
    if n == -1: # I add this case to have the linear option available
        return mu 
    else:
        s = np.sqrt(diag_Sigma)
        x = mu/s
        return nu_recursive(n,mu, diag_Sigma,x,s)

def delta (i,j):
    if i==j:
        return 1.0
    else:
        return 0.0
        
# Elementwise calculation of nu (harcoded for n = 2)
def elementwise_harcoded(mu, Sigma,W): 
    s = np.empty([N])
    x = np.empty([N])
    nu = np.empty([N])
    gamma = np.empty([N])
    Gamma = np.empty([N,N])
    J = np.empty([N,N])
    
    print(k)
    
    # Auxiliary vars
    for i in range(N):
        s[i] = np.sqrt(Sigma[i,i])
        x[i] = mu[i]/s[i]
        nu[i] = k * ( (mu[i]*mu[i]+Sigma[i,i])*f2_exact(x[i]) + mu[i]*s[i]*f1(x[i]) )
        gamma[i] = 2*k* ( mu[i]*f2_exact(x[i]) + s[i]*f1(x[i]) ) 
    
    for i in range(N):
        for j in range(N):
            Gamma[i][j] = 2*k*Sigma[i,j] * ( mu[j]*f2_exact(x[j]) + s[j]*f1(x[j]) )
            J[i][j] = tau_inv[i][0] * ( W[i][j]*gamma[j] - delta(i,j))        
    
    return (nu,gamma,Gamma,J)    

# Recursive definition of nu in terms of the power of the non linearity
def nu_recursive(m,mu,diag_Sigma,x,s): 
    if m == 0: # Adding this case allows to avoid the two cases for Gamma
        return k*f2_exact(x) 
    elif m == 1:
        return k*(mu*f2_exact(x)+s*f1(x))
    elif m == 2:
        return (mu*nu_recursive(1,mu,diag_Sigma,x,s)+
                      k * diag_Sigma * f2_exact(x))
    elif m > 2:
        return (mu*nu_recursive(m-1,mu,diag_Sigma,x,s)+
                      (m-1)* diag_Sigma * nu_recursive(m-2,mu,diag_Sigma,x,s))

# Return cov Gamma = <u_i r_j>
def get_Gamma(mu, Sigma, diag_Sigma): # Return cov. mat. <r*u> 
    if n == -1: # I add this case to have the linear option available
        return Sigma
    elif n>0:
        s = np.sqrt(diag_Sigma)
        x = mu/s
        gamma = get_gamma(mu, diag_Sigma)
        
    return Sigma * gamma
        
# Return the input noise covariance Sigma*

def get_Sigma_star(noise_scale,Sigma_x,J):
    M_inv = np.linalg.inv(tau_n_inv_id_N - np.transpose(J))
    return np.dot(noise_scale * Sigma_x, tau_inv*M_inv)

# Return mat J for the Lyap. Eqs.        
def get_J(W,gamma):  
    if n == -1:
        J = tau_inv * (W - id_N)
    else:
        J = tau_inv*(W * gamma - id_N)
    return J


# Jacobian of the original system       

def get_Jacobian(W,u):  
    if n == -1:
        J = tau_inv * (W - id_N)
    else:
        D = np.diag(k*n*np.power(0.5*(u+np.absolute(u)),n-1))
        J = tau_inv*(W @ D- id_N)
    return J
    
def get_gamma(mu, diag_Sigma):  
    if n == -1:
        gamma = 1.0
    else:
        s = np.sqrt(diag_Sigma)
        x = mu/s
        gamma = np.expand_dims(n*nu_recursive(n-1,mu, diag_Sigma,x,s),axis=0)
    return gamma
    

def get_A(m,n,mu,diag_Sigma,x,s):
    N = len(mu)
    A_mat = np.empty([N,N])
    for i in range(N):
        for j in range(N):
            if (x[i] > x[j]): (i_,j_)= (i,j)
            else: (i_,j_)= (j,i)
            A_mat[i,j] = a_ij_recursive(m,n,mu,diag_Sigma,x,s,i_,j_)
            
    return A_mat
    
def a_ij_recursive(m,n,mu,diag_Sigma,x,s,i,j):
    if (n == 0) : 
        nu_m_j = nu_recursive(m,mu[j],diag_Sigma[j],x[j],s[j])
        return k * nu_m_j
    elif (n == 1):
        nu_m_j = nu_recursive(m,mu[j],diag_Sigma[j],x[j],s[j])
        gamma_m_j = m*nu_recursive(m-1,mu[j], diag_Sigma[j],x[j],s[j])
        Gamma_m_jj = diag_Sigma[j]*gamma_m_j
        return (k*(mu[i] * nu_m_j + (s[i] /s[j]) * Gamma_m_jj))
    else:
        A_m_n_1_ij = a_ij_recursive(m,n-1,mu,diag_Sigma,x,s,i,j)
        A_m_n_2_ij = a_ij_recursive(m,n-2,mu,diag_Sigma,x,s,i,j)
        A_m_1_n_1_ij = a_ij_recursive(m-1,n-1,mu,diag_Sigma,x,s,i,j)
        return (mu[i]* A_m_n_1_ij + (n-1)* diag_Sigma[i] * A_m_n_2_ij
                                        + m * s[i]*s[j] * A_m_1_n_1_ij)

def get_B(m,n,mu,diag_Sigma,x,s):
    N = len(mu)
    B_mat = np.empty([N,N])
    for i in range(N):
        for j in range(N):
            if (x[i] > -x[j]):
                B_mat[i,j] = b_ij_recursive(m,n,mu,diag_Sigma,x,s,i,j)
            else:
                B_mat[i,j] = 0.0    
            
    return B_mat
    
def b_ij_recursive(m,n,mu,diag_Sigma,x,s,i,j):
    if (n==0):
        if (m==0):
            psi_i = f2_exact(x[i])
            psi_j = f2_exact(x[j])
            return (k*k*(psi_i+psi_j-1))
        elif (m==1):
            phi_i = f1(x[i])
            phi_j = f1(x[j])
            B00 = b_ij_recursive(0,0,mu,diag_Sigma,x,s,i,j)
            return (mu[j]*B00+ k*k*s[j]*(phi_j-phi_i))
            
        else:
            phi_i = f1(x[i])
            B0m_1 = b_ij_recursive(m-1,0,mu,diag_Sigma,x,s,i,j)
            B0m_2 = b_ij_recursive(m-2,0,mu,diag_Sigma,x,s,i,j)
            Rij = mu[j] + mu[i] * s[j]/s[i]
            return (mu[j]*B0m_1 - s[j]*(k*k*phi_i*Rij**(m-1) - (m-1) * s[j] * B0m_2))
              
    elif (n==1):
        if (m ==0):
            phi_i = f1(x[i])
            phi_j = f1(x[j])
            B0m = b_ij_recursive(m,0,mu,diag_Sigma,x,s,i,j)              
            return (mu[i]*B0m + k*k*s[i]*(phi_i-phi_j))
            
        else:
            phi_i = f1(x[i])
            B0m = b_ij_recursive(m,0,mu,diag_Sigma,x,s,i,j)
            B0m_1 = b_ij_recursive(m-1,0,mu,diag_Sigma,x,s,i,j)
            Rij = mu[j] + mu[i] * s[j]/s[i]
            return (mu[i]*B0m + s[i] *(k*k*phi_i*Rij**m - m * s[j] * B0m_1))
              
    else:
        B_m_n_1 = b_ij_recursive(m,n-1,mu,diag_Sigma,x,s,i,j)  
        B_m_n_2 = b_ij_recursive(m,n-2,mu,diag_Sigma,x,s,i,j)
        B_m_1_n_1 = b_ij_recursive(m-1,n-1,mu,diag_Sigma,x,s,i,j)
        return (mu[i]* B_m_n_1 + (n-1)*diag_Sigma[i]*B_m_n_2 - m * s[i] * s[j]* B_m_1_n_1)
                                      
def get_Lambda_plus(mu,nu,diag_Sigma,x,s):
    return (get_A(n,n,mu,diag_Sigma,x,s)-np.outer(nu,nu))
    
def get_Lambda_minus(mu,nu,diag_Sigma,x,s):
    return (get_B(n,n,mu,diag_Sigma,x,s)-np.outer(nu,nu))
    
def get_Lambda(mu,nu,Sigma, diag_Sigma):
    N = len(mu)
    
    s = np.sqrt(diag_Sigma)
    x = mu/s
    
    Lambda = np.empty([N,N])
    L_plus = get_Lambda_plus(mu,nu,diag_Sigma,x,s)
    L_minus = get_Lambda_minus(mu,nu,diag_Sigma,x,s)
    
    gamma_s = n*nu_recursive(n-1,mu, diag_Sigma,x,s)*s
    
    alpha1 = np.outer(gamma_s,gamma_s)
    alpha2 = 0.5*(L_plus + L_minus)
    alpha3 = 0.5*(L_plus - L_minus) - alpha1
    
    for i in range(N):
        for j in range(N):
            cij = Sigma[i,j]/(s[i]*s[j])
            Lambda[i,j] = alpha3[i,j] * cij**3 + alpha2[i,j] * cij**2 + alpha1[i,j] * cij
    
    return Lambda

def get_fano_factor(nu,Lambda,tau_A,t_win):
    diag_Lambda = np.diag(Lambda)
    c = 1.0 - (tau_A/t_win)*(1.0- np.exp(-t_win/tau_A))
    return 1.0 + 2.0 * tau_A * c * diag_Lambda / nu 
           
# Mat[alpha] = Chol[alpha] * Chol[alpha]^T
def Mat_from_Chol(Chol): 
    return Chol @ Chol.T


##############################################
#   Noise Buffer (lagged noise input case)   #
##############################################

def buffer_init(buffer_size,L):
    noise_init = np.reshape(np.random.normal(loc=0.0, scale=1.0, size=N*buffer_size),[buffer_size,N])
    return noise_init @ L.T
    
def update_buffer(old_bufer,new_array):
    return np.concatenate((np.resize(new_array,[1,new_array.size]), old_bufer[0:-1,:]),0)

    
###########################
#   Evolution equations   #
###########################

def new_u(u,r,h,W,eta):
    return u + dt * np.squeeze(tau_inv) * (-u + h + W @ r + eta)

def u_dot_det(W,h,u):
    return np.squeeze(tau_inv) * (-u + h + W @ get_r(u))

def new_u_det_Euler(W,h,u):
    return u + dt * u_dot_det(W,h,u)

def new_u_det_4RK(W,h,u): # Fourth order Runge-Kutta implementation
    k1 = u_dot_det(W,h,u)
    u1 = u + 0.5 * dt * k1
    k2 = u_dot_det(W,h,u1)
    u2 = u + 0.5 * dt * k2
    k3 = u_dot_det(W,h,u2)
    u3 = u + dt * k3
    k4 = u_dot_det(W,h,u3)   
    return u + (dt/6.0)*(k1+2*k2+2*k3+k4)

def new_u_det_4RK_reverse(W,h,u): # Fourth order Runge-Kutta implementation
    dt_ = -dt
    k1 = u_dot_det(W,h,u)
    u1 = u + 0.5 * dt_ * k1
    k2 = u_dot_det(W,h,u1)
    u2 = u + 0.5 * dt_ * k2
    k3 = u_dot_det(W,h,u2)
    u3 = u + dt_*k3
    k4 = u_dot_det(W,h,u3)   
    return u + (dt_/6.0)*(k1+2*k2+2*k3+k4)

def new_u_disc(u,h,eta):
    return u + dt* np.squeeze(tau_inv)*(h+eta-u)

def mu_dot(W,h,mu,nu):
    return np.squeeze(tau_inv) * (-mu + h + W @ nu)
    
def Sigma_dot(Sigma,Sigma_x,noise_scale, J, Sigma_Star = zero_Mat):
    if white:
        B = noise_scale*Sigma_x     # B: inhomogeneous term for the Lyap. Eq.
    else:
        T_inv_S_star = np.multiply(tau_inv,Sigma_Star)
        B = T_inv_S_star + T_inv_S_star.T
    
    J_Sigma = J @ Sigma
    return B + J_Sigma + J_Sigma.T

def new_mu_Euler(W,h,mu,diag_Sigma): # Simple Euler integration
    return mu + dt * mu_dot(W,h,mu,get_nu(mu, diag_Sigma))

def mu_dot_elementwise(W,h,mu,nu):
    dot = np.empty([N])
    for i in range(N):
        sum = 0.0
        for j in range(N):
            sum += W[i][j] * nu[j]
        dot[i] = tau_inv[i][0]*((h[i]-mu[i])+sum)
    return dot

def new_mu_4RK(W,h,mu,diag_Sigma): # Fourth order Runge-Kutta implementation
    k1 = mu_dot(W,h,mu,get_nu(mu, diag_Sigma))
    mu1 = mu + 0.5 * dt * k1
    k2 = mu_dot(W,h,mu1,get_nu(mu1, diag_Sigma))
    mu2 = mu + 0.5 * dt * k2
    k3 = mu_dot(W,h,mu2,get_nu(mu2, diag_Sigma))
    mu3 = mu + dt * k3
    k4 = mu_dot(W,h,mu3,get_nu(mu3, diag_Sigma))        
    return mu + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def new_Sigma_star(Sigma_x,noise_scale,Sigma_star,J):
    Sigma_Star_JT = Sigma_star @ J.T
    tmp1 = Sigma_star  + dt  * Sigma_Star_JT
    tmp2 = noise_scale * (Sigma_x * tau_inv.T) 
    return eps1 * tmp1 + eps2 * tmp2
    
def new_Sigma_star_first_order(Sigma_x,noise_scale,Sigma_star,J):
    Sigma_Star_JT = Sigma_star @ J.T
    dSigma_star = Sigma_star @ (-tau_n_inv_id_N + J.T) + noise_scale * Sigma_x * tau_inv.T
    return Sigma_star + dt * dSigma_star

def new_Sigma(Sigma,Sigma_x,noise_scale, J, Sigma_Star = zero_Mat):
    # dS = AS + SA^t+B
    # S_new = (I+dtA)S_old(I+dtA)^t+dtB
    
    if white:
        B = noise_scale*Sigma_x     # B: inhomogeneous term for the Lyap. Eq.
    else:
        T_inv_S_star = tau_inv * Sigma_Star
        B1 = T_inv_S_star + T_inv_S_star.T
        S_star_J_T = Sigma_Star @ J.T
        B2 = dt * (S_star_J_T + S_star_J_T.T)
        B = B1 + B2
    I_plus_dtJ = id_N + dt*J
    return I_plus_dtJ @ (Sigma @ I_plus_dtJ.T) + dt*B

def network_evolution(W,h,u,Sigma_eta,steps_max = 50000,eta = 0.0):
    steps = 0
    
    u_old = np.copy(u)
    eta_old = np.copy(eta)
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    
    while steps < steps_max:
        steps += 1
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta_new = eps_1 * eta_old + eps_2 * temp
        u_new = new_u(u_old,r,h,W,eta_old)
        u_old = np.copy(u_new)
        eta_old = np.copy(eta_new)
        if (np.linalg.norm(u_new) > 1000):
            print("Activity is exploding. Step: ", steps)
    print("Net evolved after "+str(steps)+" steps")
    return (u_new, eta_new)

def deterministic_network_evolution(W,h,u,steps_max = 50000):
    steps = 0
    u_new = np.copy(u)
    while steps < steps_max:
        steps += 1
        #u_new = new_u_det_Euler(W,h,u_new)
        u_new = new_u_det_4RK(W,h,u_new)
        if (np.linalg.norm(u_new) > 1000):
            print("Activity is exploding. Step: ", steps)
    print("Net evolved after "+str(steps)+" steps")
    return (u_new)

def linear_network_evolution(J_lin,u,Sigma_eta,steps_max = 50000,eta = 0.0,u_0 = zero_vec, u_dot_0 = zero_vec):
    steps = 0
    
    u_old = np.copy(u)
    eta_old = np.copy(eta)
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    
    while steps < steps_max:
        steps += 1
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta_new = eps_1 * eta_old +  eps_2 * temp
        u_new = u_old + dt * (u_dot_0 + J_lin @ (u_old-u_0) 
                                + np.squeeze(tau_inv) * eta_old)
        u_old = np.copy(u_new)
        eta_old = np.copy(eta_new)
        if (np.linalg.norm(u_new) > 1000):
            print("Activity is exploding. Step: ", steps)
    print("Net evolved after "+str(steps)+" steps")
    return (u_new, eta_new)

def disc_network_evolution(h,u,Sigma_eta,steps_max = 50000,eta = 0.0):
    steps = 0
    
    u_old = np.copy(u)
    eta_old = np.copy(eta)
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    
    while steps < steps_max:
        steps += 1
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta_new = eps_1 * eta_old + eps_2 * temp
        u_new = new_u_disc(u_old,h,eta_old)
        u_old = np.copy(u_new)
        eta_old = np.copy(eta_new)
        if (np.linalg.norm(u_new) > 1000):
            print("Activity is exploding. Step: ", steps)
    print("Net evolved after "+str(steps)+" steps")
    return (u_new, eta_new)


def network_evolution_w_pulses(W,h_on,h_off,freq,duration,u,Sigma_eta,steps_max = 50000,eta = 0.0):
    steps = 0
    
    u_old = np.copy(u)
    eta_old = np.copy(eta)
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    
    steps_between_pulses = int(1./(dt*freq))
    duration_steps = int(duration/dt)
    
    while steps < steps_max:
        steps += 1
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta_new = eps_1 * eta_old + eps_2 * temp
        if (steps % steps_between_pulses < duration_steps):
            h_t = np.copy(h_on)
        else:
            h_t = np.copy(h_off)
        u_new = new_u(u_old,r,h_t,W,eta_old)
        u_old = np.copy(u_new)
        eta_old = np.copy(eta_new)
        if (np.linalg.norm(u_new) > 1000):
            print("Activity is exploding. Step: ", steps)
    print("Net evolved after "+str(steps)+" steps")
    return (u_new, eta_new)

    
def network_evolution_w_lagged_noise(W,h,u,Sigma_eta,L,buffered_noise,steps_max = 50000,eta = 0.0):
    steps = 0
    
    u_old = np.copy(u)
    eta_old = np.copy(eta)
    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    
    while steps < steps_max:
        steps += 1
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        temp_w_lag = np.concatenate((temp[:N_exc],buffered_noise[-1,N_exc:]))
        eta_new = eps_1 * eta_old + eps_2 * temp_w_lag
        buffered_noise = update_buffer(buffered_noise,temp)
        u_new = new_u(u_old,r,h,W,eta_old)
        u_old = np.copy(u_new)
        if (np.linalg.norm(u_new) > 1000):
            print("Activity is exploding. Step: ", steps)
    print("Net evolved after "+str(steps)+" steps")
    return (u_new,eta_new,buffered_noise)

def Langevin_sample(A,B,mu,sample_size,steps_bet_samp,u0 = np.zeros(N),eta0 = np.zeros(N)):
    steps = 0
    u_samples = np.empty([sample_size,N])
    eta_samples = np.empty([sample_size,N])
    u = np.copy(u0)
    eta = np.copy(eta0)
    samples = 0
    time = 0.0
    
    Sigma_eta = tau_n_inv * B
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    steps = 0        
    while samples < sample_size:
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta = eps_1 * eta +  eps_2 * temp
        u += dt * tau_lang_inv * ( A @ (u -mu) + eta)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u
            eta_samples[samples] = eta
            samples += 1
            if (samples %5000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    return (u_samples,eta_samples,u,eta)



def network_sample(W,h,u0,eta0,sample_size,steps_bet_samp,Sigma_eta):
    steps = 0
    u_samples = np.empty([sample_size,N])
    eta_samples = np.empty([sample_size,N])
    u_old = np.copy(u0)
    eta_old = np.copy(eta0)
    samples = 0
    time = 0.0
    
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    steps = 0        
    while samples < sample_size:
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta_new = eps_1 * eta_old + eps_2 * temp
        u_new = new_u(u_old,r,h,W,eta_old)
        u_old = np.copy(u_new)
        eta_old = np.copy(eta_new)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u_new
            eta_samples[samples] = eta_new
            samples += 1
            if (samples %10000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    return (u_samples,eta_samples,u_new,eta_new)
    
def deterministic_network_sample(W,h,u0,sample_size,steps_bet_samp):
    steps = 0
    u_samples = np.empty([sample_size,N])
    u_new = np.copy(u0)
    samples = 0
    time = 0.0
    steps = 0        
    while samples < sample_size:
        #u_new = new_u_det_Euler(W,h,u_new)
        u_new = new_u_det_4RK(W,h,u_new)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u_new
            samples += 1
            if (samples %5000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    return (u_samples,u_new)

def deterministic_network_sample_reverse(W,h,u0,sample_size,steps_bet_samp):
    steps = 0
    u_samples = np.empty([sample_size,N])
    u_new = np.copy(u0)
    samples = 0
    time = 0.0
    steps = 0
    warn = False        
    while samples < sample_size:
        if ((np.linalg.norm(u_new)/np.sqrt(N))> 1000):
            warn = True
            u_new = u_new
        else:    
            u_new = new_u_det_4RK_reverse(W,h,u_new)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u_new
            samples += 1
            if (samples %5000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    if (warn): print("Activity was exploding")
    return (u_samples,u_new)
    
def linear_network_sample(J_lin,u0,eta0,sample_size,steps_bet_samp,Sigma_eta,u_0 = zero_vec,u_dot_0 = zero_vec):
    steps = 0
    u_samples = np.empty([sample_size,N])
    eta_samples = np.empty([sample_size,N])
    u_old = np.copy(u0)
    eta = np.copy(eta0)
    samples = 0
    time = 0.0
    
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    steps = 0        
    while samples < sample_size:
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta = eps_1 * eta + eps_2 * temp
        u_new = u_old + dt * (u_dot_0 + J_lin @ (u_old-u_0) + np.squeeze(tau_inv) * eta)
        u_old = np.copy(u_new)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u_new
            eta_samples[samples] = eta
            samples += 1
            if (samples %5000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    return (u_samples,eta_samples,u_new,eta)


def disc_network_sample(h,u0,eta0,sample_size,steps_bet_samp,Sigma_eta):
    steps = 0
    u_samples = np.empty([sample_size,N])
    eta_samples = np.empty([sample_size,N])
    u_old = np.copy(u0)
    eta = np.copy(eta0)
    samples = 0
    time = 0.0
    
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    steps = 0        
    while samples < sample_size:
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta = eps_1 * eta + eps_2 * temp
        u_new = new_u_disc(u_old,h,eta)
        u_old = np.copy(u_new)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u_new
            eta_samples[samples] = eta
            samples += 1
            if (samples %5000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    return (u_samples,eta_samples,u_new,eta)

def network_sample_w_pulses(W,h_on,h_off,freq,duration,u0,eta0,sample_size,steps_bet_samp,Sigma_eta):
    
    u_samples = np.empty([sample_size,N])
    eta_samples = np.empty([sample_size,N])
    u_old = np.copy(u0)
    eta = np.copy(eta0)
    
    time = 0.0
    
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    
    steps_between_pulses = int(1./(dt*freq))
    duration_steps = int(duration/dt)
    
    steps = 0
    samples = 0       
    while samples < sample_size:
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta = eps_1 * eta + eps_2 * temp
        if (steps % steps_between_pulses < duration_steps):
            h_t = np.copy(h_on)
        else:
            h_t = np.copy(h_off)   
        u_new = new_u(u_old,r,h_t,W,eta)
        u_old = np.copy(u_new)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u_new
            eta_samples[samples] = eta
            samples += 1
            if (samples %5000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    return (u_samples,eta_samples,u_new,eta)


def network_sample_w_delay(W,h0,h1,delay_times,u0,eta0,sample_size,steps_bet_samp,Sigma_eta):
    steps = 0
    u_samples = np.empty([sample_size,N])
    eta_samples = np.empty([sample_size,N])
    u_old = np.copy(u0)
    eta_old = np.copy(eta0)
    samples = 0
    time = 0.0
    
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    steps = 0        
    h_in = np.copy(h0)
    while samples < sample_size:
        arrived = np.where(delay_times <= time)
        h_in[arrived] = h1[arrived]
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta_new = eps_1 * eta_old + eps_2 * temp
        u_new = new_u(u_old,r,h_in,W,eta_old)
        u_old = np.copy(u_new)
        eta_old = np.copy(eta_new)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u_new
            eta_samples[samples] = eta_new
            samples += 1
            if (samples %5000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    return (u_samples,eta_samples,u_new,eta_new)

    
def network_sample_w_lagged_noise(W,h,u0,eta0,sample_size,steps_bet_samp,Sigma_eta,L,buffered_noise):
    steps = 0
    u_samples = np.empty([sample_size,N])
    eta_samples = np.empty([sample_size,N])
    u_old = np.copy(u0)
    eta = np.copy(eta0)
    samples = 0
    time = 0.0
    
    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    steps = 0        
    while samples < sample_size:
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        temp_w_lag = np.concatenate((temp[:N_exc],buffered_noise[-1,N_exc:]))
        eta = eps_1 * eta + eps_2 * temp_w_lag
        buffered_noise = update_buffer(buffered_noise,temp)
        u_new = new_u(u_old,r,h,W,eta)
        u_old = np.copy(u_new)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u_new
            eta_samples[samples] = eta
            samples += 1
            if (samples %5000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    return (u_samples,eta_samples,u_new,eta,buffered_noise)
    
def network_sample_w_transient_overshoot(W,h,baseline,u0,eta0,sample_size,steps_bet_samp,Sigma_eta):
    steps = 0
    u_samples = np.empty([sample_size,N])
    u_old = np.copy(u0)
    eta = np.copy(eta0)
    samples = 0
    time = 0.0
    
    h_ = h - baseline
    
    L = np.linalg.cholesky(Sigma_eta)

    eps_1 = (1.0-dt*tau_n_inv)
    eps_2 = np.sqrt(2.0*dt*tau_n_inv)
    steps = 0        
    while samples < sample_size:
        r = get_r(u_old)
        temp = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta = eps_1 * eta + eps_2 * temp
        h_mod = baseline + h_ * h_modulation(time)
        u_new = new_u(u_old,r,h_mod,W,eta)
        u_old = np.copy(u_new)
        time += dt
        if (steps % steps_bet_samp == 0):
            u_samples[samples] = u_new
            samples += 1
            if (samples %5000 == 0): print(str(samples) + " samples taken")
        steps += 1      
    return (u_samples,u_new,eta)

def mimic_cost_evolution(W,h,mu_tgt,Sigma_tgt,Sigma_x,noise_scale,
                        steps_max,steps_start_count,steps_subsamp,
                        lambda_mean,lambda_var,lambda_cov,alpha):
    
    steps = 0
    h_in = np.copy(h[alpha])
    mu_old = np.zeros(N)
    Sigma_old = 4.0 * id_N
    Sigma_star_old = 4.0 * id_N 
    
    finish = False
    
    cost_mean = 0.0
    cost_var = 0.0
    cost_cov = 0.0
    
    while (steps < steps_max):
        
        diag_Sigma_old = np.diag(Sigma_old)
        gamma = get_gamma(mu_old, diag_Sigma_old)
        J = get_J(W,gamma)
        mu_new = new_mu_Euler(W,h_in,mu_old,diag_Sigma_old)
        
        Sigma_new = new_Sigma(Sigma_old,Sigma_x,noise_scale,J,Sigma_star_old)
        Sigma_star_new = new_Sigma_star(Sigma_x,noise_scale,Sigma_star_old,J)
        
        for i in range(len(Sigma_new)):
            if Sigma_new[i][i] < 0.0:
                finish = True
                print("Element ", (i,i), " has flipped.")
        if finish == True:
            print("Finishing evolution on step ", steps)
            return (mu_new,Sigma_new,Sigma_star_new,cost_mean,cost_var,cost_cov)
        
        mu_old = np.copy(mu_new)
        Sigma_old = np.copy(Sigma_new)
        Sigma_star_old = np.copy(Sigma_star_new)
        
        steps += 1
        
        if ((steps > steps_start_count) and (steps%steps_subsamp==0)):
            cost_mean += lambda_mean * sqr_norm(mu_new[:N_exc]-mu_tgt[alpha])
            cost_var += lambda_var * sqr_norm(np.diag(Sigma_new[:N_exc,:N_exc])-np.diag(Sigma_tgt[alpha]))
            cost_cov += lambda_cov * sqr_norm(Sigma_new[:N_exc,:N_exc]-Sigma_tgt[alpha])
        
    return (mu_new,Sigma_new,Sigma_star_new,cost_mean,cost_var,cost_cov)



def slowness_cost(W,mu,Sigma,
                steps_max_slow,
                lambda_slow,alpha):
    
    steps = 0
    mu_old = np.copy(mu[alpha])
    Sigma_old = np.copy(Sigma[alpha])
    
    cost_slow = 0.0
    
    diag_Sigma_old = np.diag(Sigma_old)
    gamma = get_gamma(mu_old, diag_Sigma_old)
    J = get_J(W,gamma)
    J_T = J.T
    
    S_old = np.copy(Sigma_old)
    
    while (steps < steps_max_slow):
        cost_slow += sqr_norm(np.diag(S_old)/np.diag(Sigma_old))
        S_new = S_old + dt * (S_old @ J_T)
        S_old = np.copy(S_new)
        steps += 1
    
    cost_slow *= lambda_slow
        
    return cost_slow


def moment_evolution(location,W,h,mu,Sigma,Sigma_x,noise_scale,alpha):
    steps_max = int(20/dt)
    steps_min = steps_max/25
    w_scale = max(np.amax(np.absolute(W)),0.001)
    d_min = dt* w_scale*1e-9
        
    f = open(location+'/moment_evolution_'+str(alpha), 'w')

    steps = 0
    h_in = np.copy(h[alpha])
    if (mu.size == N) and (Sigma.size == N*N):
        mu_old = np.copy(mu)
        Sigma_old = np.copy(Sigma)
    elif (mu.size == N*N_pat) and (Sigma.size == N*N*N_pat):
        mu_old = np.copy(mu[alpha])
        Sigma_old = np.copy(Sigma[alpha])
    else:
        print("Wrong array size for the initial conditions")
        sys.exit()
        
    if not white: Sigma_star_old = 4.0 * id_N 
        
    dm = 1.0
    ds = 1.0
    
    print(0.0, np.linalg.norm(mu_old)/(1.0*np.sqrt(N)), np.linalg.norm(Sigma_old)/(1.0*N), 0.0, 0.0, file=f)
    
    
    finish = False
    
    while ((dm >= d_min or ds >= d_min) and steps < steps_max) or (steps < steps_min):
        
        diag_Sigma_old = np.diag(Sigma_old)
        gamma = get_gamma(mu_old, diag_Sigma_old)
        J = get_J(W,gamma)
    
        mu_new = new_mu_Euler(W,h_in,mu_old,diag_Sigma_old)
        
        if white:
            Sigma_new = new_Sigma(Sigma_old,Sigma_x,noise_scale,J)
        else:
            Sigma_new = new_Sigma(Sigma_old,Sigma_x,noise_scale,J,Sigma_star_old)
            Sigma_star_new = new_Sigma_star(Sigma_x,noise_scale,Sigma_star_old,J)
            
        for i in range(len(Sigma_new)):
            if Sigma_new[i][i] < 0.0:
                finish = True
                print("Element ", (i,i), " has flipped.")
        if finish == True:
            print("Finishing evolution on step ", steps)
            return (mu_old,Sigma_old,Sigma_star_old)
        
        dm = np.average(np.absolute(np.add(mu_new,-mu_old)))
        ds = np.average(np.absolute(np.add(Sigma_new,-Sigma_old)))
        mu_old = np.copy(mu_new)
        Sigma_old = np.copy(Sigma_new)
        Sigma_star_old = np.copy(Sigma_star_new)
        
        if (steps%50==0):
            print(steps*dt, np.linalg.norm(mu_new)/(1.0*np.sqrt(N)), np.linalg.norm(Sigma_new)/(1.0*N), dm, ds, file=f)
        steps += 1
    
    print(steps*dt, np.linalg.norm(mu_new)/(1.0*np.sqrt(N)), np.linalg.norm(Sigma_new)/(1.0*N), dm, ds, file=f)
    f.close()
    
    print("Convergence took: ", steps, " steps. Last steps: ", dm, ds)
    return (mu_new,Sigma_new,Sigma_star_new)
    

def moment_sample(W,h,mu,Sigma,Sigma_x,noise_scale,sample_time):
    steps_max = int(sample_time/dt)
    
    steps = 0
    h_in = h[0]
    
    mu_old = mu[0]
    Sigma_old = Sigma[0]
    
    mu_sample = np.empty([steps_max,N])
    Sigma_sample = np.empty([steps_max,N,N])
    mu_sample[0] = mu_old
    Sigma_sample[0] = Sigma_old

    finish = False
    
    Sigma_noise = get_Sigma_noise(noise_scale,Sigma_x)
        
    while steps < (steps_max-1):
        steps += 1
        diag_Sigma = np.diag(Sigma_old)
        nu = get_nu(mu_old, diag_Sigma)
        mu_new = new_mu(W,h_in,mu_old,nu)
        Sigma_new = new_Sigma(W,mu_old,Sigma_old,Sigma_noise)
        for i in range(len(Sigma_new)):
            if Sigma_new[i][i] < 0.0:
                finish = True
                print("Element ", (i,i), " has flipped.")
        if finish == True:
            print("Finishing evolution on step ", steps)
            return (mu_sample,Sigma_sample)
        mu_old = np.copy(mu_new)
        Sigma_old = np.copy(Sigma_new)
        mu_sample[steps] = mu_new
        Sigma_sample[steps] = Sigma_new
        
    return (mu_sample,Sigma_sample)

def moment_evolution_linear(J_lin,mu,Sigma,Sigma_x,noise_scale, mu_0 = zero_vec, mu_dot_0 = zero_vec):
    steps_max = int(20/dt)
    steps_min = steps_max/25
    d_min = dt* 1e-11
    
    mu_old = np.copy(mu)
    Sigma_old = np.copy(Sigma)
    if not white: Sigma_star_old = 4.0 * id_N 
    
    dm = 1.0
    ds = 1.0
    
    steps = 0
    finish = False
    
    while ((dm >= d_min or ds >= d_min) and steps < steps_max) or (steps < steps_min):
        
        mu_new = mu_old + dt * ( mu_dot_0 + J_lin @ (mu_old-mu_0))
        
        if white:
            Sigma_new = new_Sigma(Sigma_old,Sigma_x,noise_scale,J_lin)
        else:
            Sigma_new = new_Sigma(Sigma_old,Sigma_x,noise_scale,J_lin,Sigma_star_old)
            Sigma_star_new = new_Sigma_star(Sigma_x,noise_scale,Sigma_star_old,J_lin)
        
        for i in range(len(Sigma_new)):
            if Sigma_new[i][i] < 0.0:
                finish = True
                print("Element ", (i,i), " has flipped.")
        if finish == True:
            print("Finishing evolution on step ", steps)
            return (mu_old,Sigma_old,Sigma_star_old)
        
        dm = np.average(np.absolute(np.add(mu_new,-mu_old)))
        ds = np.average(np.absolute(np.add(Sigma_new,-Sigma_old)))
        mu_old = np.copy(mu_new)
        Sigma_old = np.copy(Sigma_new)
        Sigma_star_old = np.copy(Sigma_star_new)
        steps += 1
    
    print("Convergence took: ", steps, " steps. Last steps: ", dm, ds)
    
    return (mu_new,Sigma_new,Sigma_star_new)

##############################
#   Simulate spike process   #
##############################


def Gamma_spike_train_time_resc(shape, rate_trace, delta_t):

    T = delta_t * np.sum(rate_trace)
    
    n_steps = max(5,np.ceil(1.5*T).astype(int))
    mean = 1.0
    scale = mean/shape
    
    spike_train = np.zeros(len(rate_trace))
    
    # we first randomize the start time
    
    start_time = np.random.rand()
    
    spike_times = np.random.gamma(shape, scale, size=n_steps)
    tot_time = np.sum(spike_times)
    
    while (tot_time<start_time):
        new_times = np.random.gamma(shape, scale, size=n_steps)
        tot_time += np.sum(new_times)
        spike_times = np.concatenate((spike_times,new_times))    
    
    done = False
    i = 0
    rest = start_time
    while not done:
        if (spike_times[i] >= rest):
            spike_times[i] -= rest
            done = True
        else:
            rest -= spike_times[i]
            spike_times[i] = 0.0
            i += 1
             
    spike_times = spike_times[i:]
    
    # now we make sure we have a long enough train
     
    tot_time = np.sum(spike_times)
    
    while (tot_time < T):
        new_times = np.random.gamma(shape, scale, size=n_steps)
        tot_time += np.sum(new_times)
        spike_times = np.concatenate((spike_times,new_times))
    
    # now we construct the vector of spikes
    
    i =0
    n_spikes = 0
    elapsed = 0
    finished = False
    while (i < len(rate_trace)):
        time = spike_times[n_spikes]
        while (elapsed < time):
            elapsed += delta_t * rate_trace[i]
            i += 1
            if (i == len(rate_trace)):
                finished = True
                break
        if (not finished):       
            spike_train[i-1] +=1
            n_spikes += 1
            elapsed -= time 
    
    return spike_train



############################
#   Evaluate Performance   #
############################

def rel_error(dx,x):
    return np.linalg.norm(dx)/np.linalg.norm(x)
