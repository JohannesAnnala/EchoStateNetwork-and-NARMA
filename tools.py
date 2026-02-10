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

def entanglement(operator):
    return np.log2(trace_norm(operator))

def assess_dm_entanglement(rho, part, dims_first, dims_second, rounding=None):
    if rounding:
        return np.round(entanglement(partial_transpose(rho, part, dims_first, dims_second)),rounding)
    return entanglement(partial_transpose(rho, part, dims_first, dims_second))

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
def init_sq_th(alpha, mean_n, truncate, a1, a2, rounding=None):
    
    #Initialize the two-mode thermal state
    sq_ = init_two_mode_sq(alpha, a1, a2)
    th_ = init_two_mode_th(mean_n, truncate)

    if rounding:  
        return truncate_mantissa(sq_ @ th_ @ dagger(sq_), rounding)
    
    return sq_ @ th_ @ dagger(sq_)

def init_sq_add(alpha, truncate, a1, a2, rounding=None):

    sq_ = init_two_mode_sq(alpha, a1, a2)
    vac_ = init_vac(truncate**2)
    sq_add_ = dagger(a1) @ dagger(a2) @ sq_ @ vac_ @ dagger(sq_) @ a2 @ a1

    #Normalize
    sq_add_ = sq_add_ / np.trace(sq_add_)

    if rounding:  
        return truncate_mantissa(sq_add_, rounding)
    
    return sq_add_

def init_sq_sub(alpha, truncate, a1, a2, rounding=None):

    sq_ = init_two_mode_sq(alpha, a1, a2)
    vac_ = init_vac(truncate**2)
    sq_sub_ = a1 @ a2 @ sq_ @ vac_ @ dagger(sq_) @ dagger(a2) @ dagger(a1)

    #Normalize
    sq_sub_ = sq_sub_ / np.trace(sq_sub_)

    if rounding:  
        return truncate_mantissa(sq_sub_, rounding)
    
    return sq_sub_

def init_simple(c0, c1, truncate, rounding=None):

    vac_site_ = np.zeros((truncate**2,), dtype=np.complex64)
    vac_site_[0] = c0
    exit_site_ = np.zeros((truncate**2,), dtype=np.complex64)
    exit_site_[6] = c1
    simple_ = np.outer(vac_site_ + exit_site_, np.conj(vac_site_ + exit_site_))

    if rounding:
        return truncate_mantissa(simple_, rounding)
    
    return simple_

#Creates two-mode input states in bulk
def gen_input_states(type, amount_of_states, truncate, rounding=None):

    #Squeezed thermal states
    if type == "sq_th":
        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)])

        theta_sq_th_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        s_sq_th_ = np.random.uniform(0.8,0.95,(amount_of_states,))
        phi_sq_th_ = np.random.uniform(0.5-np.pi/10, 0.5+np.pi/10, (amount_of_states,))
        alpha_sq_th_ = np.array([x*np.sin(y)*np.exp(1j*z) for x, y, z in zip(s_sq_th_,phi_sq_th_,theta_sq_th_)])
        mean_n_sq_th_ = np.array([x*x*np.cos(y)*np.cos(y) for x, y in zip(s_sq_th_,phi_sq_th_)])

        return np.array([init_sq_th(x,y,truncate,a1_,a2_,rounding) for x,y in zip(alpha_sq_th_,mean_n_sq_th_)])
    
    #Photon added vacuum states
    elif type == "pho_add":
        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)])
        
        abs_alpha_pho_add_ = np.random.uniform(0.1, 0.25, (amount_of_states,))
        theta_pho_add_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_pho_add_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_pho_add_,theta_pho_add_)])  

        return np.array([init_sq_add(x,truncate,a1_,a2_,rounding) for x in alpha_pho_add_])

    #Photon subtracted vacuum states
    elif type == "pho_sub":
        a1_ = tensor([init_destroy(truncate), init_identity(truncate)])
        a2_ = tensor([init_identity(truncate), init_destroy(truncate)])
        
        abs_alpha_pho_sub_ = np.random.uniform(0.8, 0.95, (amount_of_states,))
        theta_pho_sub_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        alpha_pho_sub_ = np.array([x*np.exp(1j*y) for x, y in zip(abs_alpha_pho_sub_,theta_pho_sub_)])

        return np.array([init_sq_sub(x,truncate,a1_,a2_,rounding) for x in alpha_pho_sub_])

    #Simple states 
    elif type == "simple":
        theta_simple_ = np.array([np.arcsin(np.sqrt(x)) for x in np.random.uniform(0,1,(amount_of_states,))])
        phi_simple_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        c0_simple_ = np.array([np.sin(x) for x in theta_simple_])
        c1_simple_ = np.array([np.cos(x)*np.exp(1j*y) for x,y in zip(theta_simple_, phi_simple_)])

        return np.array([init_simple(x,y,truncate,rounding) for x,y in zip(c0_simple_, c1_simple_)])
    
    else:
        print(f"{type} not possible")

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

def folder_name(gamma, res_size, fock_trunc, res_connect):
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