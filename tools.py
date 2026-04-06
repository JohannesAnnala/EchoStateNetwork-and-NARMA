import numpy as np
import scipy as sp
import copy
import os
from json import load

def dagger(operator):
    return np.conj(operator).T

def tensor(operators):
    product_ = operators[0]
    for i in range(1,len(operators)):
        product_ = np.kron(product_, operators[i])

    return product_

#Mathematics for partial trace in "Computing partial traces and reduced density matrices" by Jonas Maziero
def partial_trace(rho, part, dims_first, dims_second):
    
    if part == "first":
        rho_new_ = np.zeros((dims_first,dims_first), dtype=np.complex64)
        for i in range(dims_first):
            for j in range(dims_first):
                rho_new_[i,j] = np.trace(rho[i*dims_second:(i+1)*dims_second, j*dims_second:(j+1)*dims_second], dtype=np.complex64)
    elif part == "second":
        rho_new_ = np.zeros((dims_second,dims_second), dtype=np.complex64)
        for i in range(dims_first):
            rho_new_ += rho[i*dims_second:(i+1)*dims_second, i*dims_second:(i+1)*dims_second]
    else:
        print("Partial trace can only keep the first or second part of the system")
    
    return rho_new_

def partial_transpose(rho, part, dims_first, dims_second):  
    rho_new_ = copy.copy(rho)

    if part == "first":
        for i in range(dims_first):
            for j in range(dims_first):
                rho_new_[i*dims_second:(i+1)*dims_second,j*dims_second:(j+1)*dims_second] = rho[j*dims_second:(j+1)*dims_second,i*dims_second:(i+1)*dims_second]
    elif part == "second":    
        for i in range(dims_first):
            for j in range(dims_first):
                rho_new_[i*dims_second:(i+1)*dims_second,j*dims_second:(j+1)*dims_second] = rho[i*dims_second:(i+1)*dims_second,j*dims_second:(j+1)*dims_second].T
    else:
        print("Partial transpose can only be performed over the first or second part of the system")
        return 0

    return rho_new_

#Calculates the trace norm of a Hermitian operator
def trace_norm(operator):
    return np.sum(np.abs(np.linalg.eigvals(operator)))

def truncate_mantissa(operator, decimals):
    return np.trunc(operator.real * 10**decimals) / 10**decimals + 1j * (np.trunc(operator.imag * 10**decimals) / 10**decimals)

def real_diag(operator):
    np.fill_diagonal(operator, np.diag(operator).real)

def negativity(operator):
    return (trace_norm(operator)-1)/2

def logarithmic_negativity(operator):
    return np.log2(trace_norm(operator))

def assess_dm_entanglement(rho, part, dims_first, dims_second, rounding=None):
    if rounding:
        return truncate_mantissa(logarithmic_negativity(partial_transpose(rho, part, dims_first, dims_second)),rounding)
    return logarithmic_negativity(partial_transpose(rho, part, dims_first, dims_second))

def get_entanglement_values(dms, part, dims_first, dims_second, rounding=None):
    return np.array([[1,0] if assess_dm_entanglement(dm, "first", dims_first, dims_second, rounding) > 0 else [0,1] for dm in dms])

def expectation_value(rho, operator):
    return np.trace(rho @ operator)

def rk4(function, initial_state, t, dt):
    k1_ = function(initial_state, t)
    k2_ = function(initial_state + 0.5 * dt * k1_, t + 0.5 * dt)
    k3_ = function(initial_state + 0.5 * dt * k2_, t + 0.5 * dt)
    k4_ = function(initial_state + dt * k3_, t + dt)
    return initial_state + (dt/6)*(k1_ + 2 * k2_ + 2 * k3_ + k4_)

def data_standardize(data):
    return (data - np.mean(data, axis=0))/np.std(data, axis=0)

def init_identity(size):
    return np.eye(size)

def init_destroy(size):
    return np.diag(np.sqrt(np.arange(1,size)),1)

def init_create(size):
    return np.diag(np.sqrt(np.arange(1,size)),-1)

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

def init_sq(alpha, truncate):
    a_ = init_destroy(truncate)
    A_ = 0.5 * (alpha * dagger(a_) @ dagger(a_) - np.conjugate(alpha) * a_ @ a_)
    return sp.linalg.expm(A_)

def init_two_mode_sq(alpha, a1, a2):
    A_ = alpha * dagger(a1) @ dagger(a2) - np.conjugate(alpha) * a1 @ a2
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

def init_vac(truncate):
    vac_ = np.zeros((truncate, truncate), dtype=np.complex64)
    vac_[0,0] = 1
    return vac_

#Function that creates a two-mode squeezed thermal state
def init_ST(alpha, mean_n, truncate, a1, a2, rounding=None):
    
    #Initialize the two-mode thermal state
    sq_ = init_two_mode_sq(alpha, a1, a2)
    th_ = init_two_mode_th(mean_n, truncate)

    if rounding:  
        return truncate_mantissa(sq_ @ th_ @ dagger(sq_), rounding)
    
    return sq_ @ th_ @ dagger(sq_)

def init_PASV(alpha, truncate, a1, a2, rounding=None):

    sq_ = init_two_mode_sq(alpha, a1, a2)
    vac_ = init_vac(truncate**2)
    sq_add_ = dagger(a1) @ dagger(a2) @ sq_ @ vac_ @ dagger(sq_) @ a2 @ a1

    #Normalize
    sq_add_ = sq_add_ / np.trace(sq_add_)

    if rounding:  
        return truncate_mantissa(sq_add_, rounding)
    
    return sq_add_

def init_PASV_sep(alpha1, alpha2, truncate, a, rounding=None):

    sq1_ = init_sq(alpha1, truncate)
    sq2_ = init_sq(alpha2, truncate)
    vac_ = init_vac(truncate)
    sq_add1_ = dagger(a) @ sq1_ @ vac_ @ dagger(sq1_) @ a   
    sq_add2_ = dagger(a) @ sq2_ @ vac_ @ dagger(sq2_) @ a

    #Normalize
    sq_add1_ = sq_add1_ / np.trace(sq_add1_)    
    sq_add2_ = sq_add2_ / np.trace(sq_add2_)

    if rounding:
        return truncate_mantissa(tensor([sq_add1_, sq_add2_]), rounding)
    
    return tensor([sq_add1_, sq_add2_])

def init_PSSV(alpha, truncate, a1, a2, rounding=None):

    sq_ = init_two_mode_sq(alpha, a1, a2)
    vac_ = init_vac(truncate**2)
    sq_sub_ = a1 @ a2 @ sq_ @ vac_ @ dagger(sq_) @ dagger(a2) @ dagger(a1)

    #Normalize
    sq_sub_ = sq_sub_ / np.trace(sq_sub_)

    if rounding:  
        return truncate_mantissa(sq_sub_, rounding)
    
    return sq_sub_

def init_PSSV_sep(alpha1, alpha2, truncate, a, rounding=None):

    a = init_destroy(truncate)
    sq1_ = init_sq(alpha1, truncate)
    sq2_ = init_sq(alpha2, truncate)
    vac_ = init_vac(truncate)
    sq_sub1_ = a @ sq1_ @ vac_ @ dagger(sq1_) @ dagger(a)   
    sq_sub2_ = a @ sq2_ @ vac_ @ dagger(sq2_) @ dagger(a)

    #Normalize
    sq_sub1_ = sq_sub1_ / np.trace(sq_sub1_)    
    sq_sub2_ = sq_sub2_ / np.trace(sq_sub2_)

    if rounding:
        return truncate_mantissa(tensor([sq_sub1_, sq_sub2_]), rounding)
    
    return tensor([sq_sub1_, sq_sub2_])

def init_NUM(c0, c1, truncate, rounding=None):

    vac_site_ = np.zeros((truncate**2,), dtype=np.complex64)
    vac_site_[0] = c0
    exit_site_ = np.zeros((truncate**2,), dtype=np.complex64)
    exit_site_[truncate+1] = c1
    NUM_ = np.outer(vac_site_ + exit_site_, np.conj(vac_site_ + exit_site_))

    if rounding:
        return truncate_mantissa(NUM_, rounding)
    
    return NUM_

def init_NUM_sep(truncate, rounding=None):
    NUM_ = np.zeros((truncate**2,truncate**2), dtype=np.complex64)
    if np.random.choice(2,1):
        NUM_[0,0] = 1
    else:
        NUM_[truncate+1,truncate+1] = 1

    if rounding:
        return truncate_mantissa(NUM_, rounding)
    
    return NUM_

#Creates two-mode input states in bulk
def gen_input_states(type, amount_of_states, truncate, entanglement=False, rounding=None):

    #Squeezed thermal states
    if type == "ST":
        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)])

        theta_ST_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        s_ST_ = np.random.uniform(0.8,0.95,(amount_of_states,))
        phi_ST_ = np.random.uniform(0.5-np.pi/10, 0.5+np.pi/10, (amount_of_states,))
        alpha_ST_ = np.array([x*np.sin(y)*np.exp(1j*z) for x, y, z in zip(s_ST_,phi_ST_,theta_ST_)])
        mean_n_ST_ = np.array([x*x*np.cos(y)*np.cos(y) for x, y in zip(s_ST_,phi_ST_)])

        ST_ = np.array([init_ST(x,y,truncate,a1_,a2_,rounding) for x,y in zip(alpha_ST_,mean_n_ST_)])

        if entanglement:
            return ST_, get_entanglement_values(ST_, "first", truncate, truncate, 2)
        return ST_
    
    #Photon added squeezed vacuum states
    elif type == "PASV":
        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)])
        
        abs_alpha_PASV_ = np.random.uniform(0.1, 0.25, (amount_of_states,))
        theta_PASV_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PASV_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PASV_,theta_PASV_)])

        PASV_ = np.array([init_PASV(x,truncate,a1_,a2_,rounding) for x in alpha_PASV_])

        if entanglement:
            return PASV_, np.array([[1,0] for _ in range(amount_of_states)])
        return PASV_

    #Separable photon added squeezed vacuum states
    elif type == "PASV_sep":
        a_ = init_destroy(truncate)
        
        abs_alpha_PASV1_ = np.random.uniform(0.1, 0.25, (amount_of_states,))
        theta_PASV1_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PASV1_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PASV1_,theta_PASV1_)])  
        abs_alpha_PASV2_ = np.random.uniform(0.1, 0.25, (amount_of_states,))
        theta_PASV2_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PASV2_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PASV2_,theta_PASV2_)])  
        
        PASV_sep_ = np.array([init_PASV_sep(x, y, truncate, a_, rounding) for x,y in zip(alpha_PASV1_, alpha_PASV2_)])    

        if entanglement:
            return PASV_sep_, np.array([[0,1] for _ in range(amount_of_states)])
        return PASV_sep_  

    elif type == "PASV_split":
        if amount_of_states % 2 != 0:
            print("For an equal split enter an even amount of states")
            return 0
        
        new_perm_ = np.random.permutation(amount_of_states)
        amount_of_states = int(amount_of_states / 2)

        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)]) 
        abs_alpha_PASV_ = np.random.uniform(0.1, 0.25, (amount_of_states,))
        theta_PASV_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PASV_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PASV_,theta_PASV_)])

        a_ = init_destroy(truncate) 
        abs_alpha_PASV1_ = np.random.uniform(0.1, 0.25, (amount_of_states,))
        theta_PASV1_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PASV1_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PASV1_,theta_PASV1_)])  
        abs_alpha_PASV2_ = np.random.uniform(0.1, 0.25, (amount_of_states,))
        theta_PASV2_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PASV2_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PASV2_,theta_PASV2_)]) 

        PASV_split_ = np.array([*[init_PASV(x,truncate,a1_,a2_,rounding) for x in alpha_PASV_], *[init_PASV_sep(x,y,truncate,a_,rounding) for x,y in zip(alpha_PASV1_,alpha_PASV2_)]])  

        if entanglement:
            entanglement_ = np.array([*[[1,0] for _ in range(amount_of_states)], *[[0,1] for _ in range(amount_of_states)]])
            return PASV_split_[new_perm_], entanglement_[new_perm_]
        return PASV_split_[new_perm_]

    #Photon subtracted squeezed vacuum states
    elif type == "PSSV":
        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)])
        
        abs_alpha_PSSV_ = np.random.uniform(0.8, 0.95, (amount_of_states,))
        theta_PSSV_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PSSV_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PSSV_,theta_PSSV_)])

        PSSV_ = np.array([init_PSSV(x,truncate,a1_,a2_,rounding) for x in alpha_PSSV_])

        if entanglement:
            return PSSV_, np.array([[1,0] for _ in range(amount_of_states)])
        return PSSV_

    #Separable photon added squeezed vacuum states
    elif type == "PSSV_sep":
        a_ = init_destroy(truncate)
        
        abs_alpha_PSSV1_ = np.random.uniform(0.8, 0.95, (amount_of_states,))
        theta_PSSV1_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PSSV1_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PSSV1_,theta_PSSV1_)])  
        abs_alpha_PSSV2_ = np.random.uniform(0.8, 0.95, (amount_of_states,))
        theta_PSSV2_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PSSV2_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PSSV2_,theta_PSSV2_)])  
        
        PSSV_sep_ = np.array([init_PSSV_sep(x, y, truncate, a_, rounding) for x,y in zip(alpha_PSSV1_, alpha_PSSV2_)])    

        if entanglement:
            return PSSV_sep_, np.array([[0,1] for _ in range(amount_of_states)])
        return PSSV_sep_  

    elif type == "PSSV_split":
        if amount_of_states % 2 != 0:
            print("For an equal split enter an even amount of states")
            return 0
        
        new_perm_ = np.random.permutation(amount_of_states)
        amount_of_states = int(amount_of_states / 2)

        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)]) 
        abs_alpha_PSSV_ = np.random.uniform(0.8, 0.95, (amount_of_states,))
        theta_PSSV_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PSSV_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PSSV_,theta_PSSV_)])

        a_ = init_destroy(truncate) 
        abs_alpha_PSSV1_ = np.random.uniform(0.8, 0.95, (amount_of_states,))
        theta_PSSV1_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PSSV1_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PSSV1_,theta_PSSV1_)])  
        abs_alpha_PSSV2_ = np.random.uniform(0.8, 0.95, (amount_of_states,))
        theta_PSSV2_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_PSSV2_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_PSSV2_,theta_PSSV2_)]) 

        PSSV_split_ = np.array([*[init_PSSV(x,truncate,a1_,a2_,rounding) for x in alpha_PSSV_], *[init_PSSV_sep(x,y,truncate,a_,rounding) for x,y in zip(alpha_PSSV1_,alpha_PSSV2_)]])  

        if entanglement:
            entanglement_ = np.array([*[[1,0] for _ in range(amount_of_states)], *[[0,1] for _ in range(amount_of_states)]])
            return PSSV_split_[new_perm_], entanglement_[new_perm_]
        return PSSV_split_[new_perm_]

    #Number states 
    elif type == "NUM":
        theta_NUM_ = np.array([0.5*np.arcsin(x) for x in np.random.uniform(0,1,(amount_of_states,))])
        phi_NUM_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        c0_NUM_ = np.array([np.sin(x) for x in theta_NUM_])
        c1_NUM_ = np.array([np.cos(x)*np.exp(1j*y) for x,y in zip(theta_NUM_, phi_NUM_)])

        NUM_ = np.array([init_NUM(x,y,truncate,rounding) for x,y in zip(c0_NUM_, c1_NUM_)])

        if entanglement:
            return NUM_, np.array([[1,0] for _ in range(amount_of_states)])
        return NUM_  
    
    elif type == "NUM_sep":

        NUM_sep_ = np.array([init_NUM_sep(truncate,rounding) for _ in range(amount_of_states)])

        if entanglement:
            return NUM_sep_, np.array([[0,1] for _ in range(amount_of_states)])
        return NUM_sep_  
    
    elif type == "NUM_split":
        if amount_of_states % 2 != 0:
            print("For an equal split enter an even amount of states")
            return 0
        
        new_perm_ = np.random.permutation(amount_of_states)
        amount_of_states = int(amount_of_states / 2)

        theta_NUM_ = np.array([0.5*np.arcsin(x) for x in np.random.uniform(0,1,(amount_of_states,))])
        phi_NUM_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        c0_NUM_ = np.array([np.sin(x) for x in theta_NUM_])
        c1_NUM_ = np.array([np.cos(x)*np.exp(1j*y) for x,y in zip(theta_NUM_, phi_NUM_)])

        NUM_split_ = np.array([*[init_NUM(x,y,truncate,rounding) for x,y in zip(c0_NUM_, c1_NUM_)], *[init_NUM_sep(truncate,rounding) for _ in range(amount_of_states)]])

        if entanglement:
            entanglement_ = np.array([*[[1,0] for _ in range(amount_of_states)], *[[0,1] for _ in range(amount_of_states)]])
            return NUM_split_[new_perm_], entanglement_[new_perm_]
        return NUM_split_[new_perm_]
    
    else:
        print(f"{type} not possible")
        return 0

def unpack_config(filepath):
    with open(filepath) as file:
        parameters = load(file)

    gamma_ = parameters["GAMMA"]                                 
    reservoir_size_ = parameters["SIZE"]    
    fock_truncation_ = parameters["TRUNC"]
    res_connect_ = parameters["CONNECT"]
    sim_rounding_ = parameters["ROUND"]
    n_models_ = parameters["MODELS"]
    n_train_inputs_ = parameters["TRAIN"]
    n_test_inputs_ = parameters["TEST"]

    return gamma_, reservoir_size_, fock_truncation_, res_connect_, sim_rounding_, n_models_, n_train_inputs_, n_test_inputs_

def gen_narma_nmse_filepath(narma_order):
    filename = "narma" + str(narma_order) + "_nmse.csv"
    return os.path.join(os.getcwd(),"reservoir_narma_nmse", filename)

def gen_config_filepath(confignumber):
    filename = "config" + str(confignumber) + ".json"
    return os.path.join(os.getcwd(),"Qreservoir_configurations", filename)

def folder_name(gamma, res_size, fock_trunc, res_connect, cm=False):
    if cm:
        return str(gamma) + "_" + str(res_size) + "_" + str(fock_trunc) + "_" + str(res_connect) + "_cm"
    return str(gamma) + "_" + str(res_size) + "_" + str(fock_trunc) + "_" + str(res_connect)

def gen_system_filepath(folder, systemnumber):
    filename = "system" + str(systemnumber) + ".npz"
    return os.path.join(os.getcwd(),"Qreservoir_systems", folder, filename)

def gen_result_filepath(filename):
    filename = filename + ".csv"
    return os.path.join(os.getcwd(),"Qreservoir_results", filename)

def write_to_row(value, filepath):
    with open(filepath, 'a') as file:
        file.write(value + ";")

def finish_row(filepath):
    with open(filepath, 'a') as file:
        file.write('0\n')