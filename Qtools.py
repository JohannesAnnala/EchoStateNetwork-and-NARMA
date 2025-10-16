import numpy as np
import scipy as sp
from tools import init_identity, dagger, tensor, negativity, partial_transpose

def init_create(size):
    return np.diag(np.sqrt(np.arange(1,size)),-1)

def init_destroy(size):
    return np.diag(np.sqrt(np.arange(1,size)),1)

def init_dissipator(rho, coperator1, coperator2=None):
    if not coperator2:
        coperator2 = coperator1

    return coperator1 @ rho @ dagger(coperator2) - 0.5 * dagger(coperator1) @ coperator2 @ rho - 0.5 * rho @ dagger(coperator1) @ coperator2 

def init_dissipators(rho, coperators):
    return sum([init_dissipator(rho, cop) for cop in coperators])

def init_multipartite_dc_operators(doperators, sizes):
    identities_ = [init_identity(x.shape[0]) for x in doperators]
    part_identities_ = [init_identity(x.shape[0] ** y) for x,y in zip(doperators, sizes)]

    doperators_ = [[tensor([*part_identities_[:idx], *x, *part_identities_[idx+1:]]) for x in \
                    [[dop if i==j else nochange for j, nochange in enumerate(site)] for i, site in enumerate([id for _ in range(size)] for _ in range(size))]] for idx, (dop, id, size) in enumerate(zip(doperators, identities_, sizes))]
    coperators_ = [[tensor([*part_identities_[:idx], *x, *part_identities_[idx+1:]]) for x in \
                    [[dagger(dop) if i==j else nochange for j, nochange in enumerate(site)] for i, site in enumerate([id for _ in range(size)] for _ in range(size))]] for idx, (dop, id, size) in enumerate(zip(doperators, identities_, sizes))]

    return doperators_, coperators_

def init_two_mode_sq(alpha, a1, a2):
    A_ = -alpha * dagger(a1) @ dagger(a2) - np.conjugate(alpha) * a1 @ a2
    return sp.linalg.expm(A_)

def init_th(mean_n, truncate, other=False):
    if other:
        beta = np.log(1.0 / mean_n + 1.0)
        values_ = np.exp(-beta * np.arange(truncate))
        values_ = values_ / np.sum(values_)
    else:
        values_ = [1/(1+mean_n)*(mean_n/(1+mean_n))**x for x in np.arange(truncate)]

    return np.diag(values_)

def init_two_mode_th(mean_n, truncate):
    return tensor([init_th(mean_n, truncate, other=True), init_th(mean_n, truncate,other=True)])

#Function that creates a two-mode squeezed thermal state
def init_two_mode_sq_th(alpha, mean_n, truncate, a1, a2):
    
    #Initialize the tmo-mode squeezing matrix and two-mode thermal state
    sq_ = init_two_mode_sq(alpha, a1, a2)
    th_ = init_two_mode_th(mean_n, truncate)

    #Return two-mode squeezed thermal state
    return sq_ @ th_ @ dagger(sq_)

#Creates two-mode input states in bulk
def gen_input_states(type, amount_of_states, truncate=None):

    #Squeezed thermal states
    if type == "sq_th":
        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)])

        theta_sq_th_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        s_sq_th_ = np.random.uniform(0.8,0.95,(amount_of_states,))
        phi_sq_th_ = np.random.uniform(0.5-np.pi/10, 0.5+np.pi/10, (amount_of_states,))
        alpha_sq_th_ = np.array([x*np.sin(y)*np.exp(1j*z) for x, y, z in zip(s_sq_th_,phi_sq_th_,theta_sq_th_)])
        mean_n_sq_th_ = np.array([x*x*np.cos(y)*np.cos(y) for x, y in zip(s_sq_th_,phi_sq_th_)])

        return np.array([init_two_mode_sq_th(x,y,truncate,a1_,a2_) for x,y in zip(alpha_sq_th_,mean_n_sq_th_)])
    
    #Photon added vacuum states
    elif type == "pho_add":
        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)])
        
        abs_alpha_pho_add_ = np.random.uniform(0.1, 0.25, (amount_of_states,))
        theta_pho_add_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_pho_add_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_pho_add_,theta_pho_add_)])  

        #return np.array([init_pho_add(x, truncate,a1_,a2_) for x in alpha_pho_add_])

    #Photon subtracted vacuum states
    elif type == "pho_sub":
        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)])
        
        abs_alpha_pho_sub_ = np.random.uniform(0.8, 0.95, (amount_of_states,))
        theta_pho_sub_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_pho_sub_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_pho_sub_,theta_pho_sub_)])

        #return np.array([init_pho_sub(x, truncate,a1_,a2_) for x in alpha_pho_sub_])

    #Simple states 
    elif type == "simple":
        theta_simple_ = np.array([np.arcsin(np.sqrt(x)) for x in np.random.uniform(0,1,(amount_of_states,))])
        phi_simple_ = np.random.uniform(0,2*np.pi,(amount_of_states,))

        c0_simple_ = np.array([np.sin(x) for x in theta_simple_])
        c1_simple_ = np.array([np.cos(x)*np.exp(1j*y) for x,y in zip(theta_simple_, phi_simple_)])

        #return np.array([init_simple(x, y, truncate) for x,y in zip(c0_simple_, c1_simple_)])
    
    else:
        print(f"{type} not possible")

def entanglement(negativity):
    return np.log2(2*negativity+1)

def assess_dm_entanglement(rho, part, dims_first, dims_second):
    return entanglement(negativity(partial_transpose(rho, part, dims_first, dims_second)))

def expectation_value(rho, operator):
    return np.trace(rho @ operator)
