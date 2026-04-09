import numpy as np
import scipy as sp
import copy
import os
from json import load

def dagger(operator):
    """
    Calculates the Hermitian adjoint of an operator.

    Parameters
    ----------
    operator : Numpy array of an operator

    Returns
    -------
    The Hermitian adjoint of operator

    """

    return np.conj(operator).T

def tensor(operators):
    """
    Performs the matrix tensor product for operators.

    Parameters
    ----------
    operators : List of Numpy arrays

    Returns
    -------
    product_ : Numpy array of the tensor product of given operators

    """

    product_ = operators[0]
    for i in range(1,len(operators)):
        product_ = np.kron(product_, operators[i])

    return product_

def partial_trace(dm, part, dims_first, dims_second):
    """
    Calculates the partial trace of a two-mode system density matrix.

    Parameters
    ----------
    dm : Numpy array of the combined density matrix
    part : String value of the system kept after the partial trace operation
    dims_first : Dimensions of the first system
    dims_second : Dimension of the second system

    Returns
    -------
    dm_new_ : Numpy array of the reduced density matrix
    
    """

    if part == "first":
        dm_new_ = np.zeros((dims_first,dims_first), dtype=np.complex64)
        for i in range(dims_first):
            for j in range(dims_first):
                dm_new_[i,j] = np.trace(dm[i*dims_second:(i+1)*dims_second, j*dims_second:(j+1)*dims_second], dtype=np.complex64)
    elif part == "second":
        dm_new_ = np.zeros((dims_second,dims_second), dtype=np.complex64)
        for i in range(dims_first):
            dm_new_ += dm[i*dims_second:(i+1)*dims_second, i*dims_second:(i+1)*dims_second]
    else:
        print("Partial trace can only keep the first or second part of the system")
    
    return dm_new_

def partial_transpose(dm, part, dims_first, dims_second):
    """
    Calculates the partial transpose of a two-mode system density matrix

    Parameters
    ----------
    dm : Numpy array of the combined density matrix
    part : String value of the system transposed after the partial trace operation
    dims_first : Dimensions of the first system
    dims_second : Dimension of the second system

    Returns
    -------
    dm_new_ : Numpy array of the partially transposed density matrix   
    
    """

    dm_new_ = copy.copy(dm)

    if part == "first":
        for i in range(dims_first):
            for j in range(dims_first):
                dm_new_[i*dims_second:(i+1)*dims_second,j*dims_second:(j+1)*dims_second] = dm[j*dims_second:(j+1)*dims_second,i*dims_second:(i+1)*dims_second]
    elif part == "second":    
        for i in range(dims_first):
            for j in range(dims_first):
                dm_new_[i*dims_second:(i+1)*dims_second,j*dims_second:(j+1)*dims_second] = dm[i*dims_second:(i+1)*dims_second,j*dims_second:(j+1)*dims_second].T
    else:
        print("Partial transpose can only be performed over the first or second part of the system")
        return 0

    return dm_new_

def trace_norm(operator):
    """
    Calculates the trace norm of a Hermitian operator.

    Parameters
    ----------
    operator : Numpy array of an operator

    Returns
    -------
    The sum of absolute eigenvalues of the operator

    """

    return np.sum(np.abs(np.linalg.eigvals(operator)))

def truncate_mantissa(operator, decimals):
    """
    Truncates the decimal mantissa of all values in an operetor without rounding.

    Parameters
    ----------
    operator : Numpy array of an operator
    decimals : Int value of the amount of decimals saved

    Returns
    -------
    The truncated Numpy array

    """

    return np.trunc(operator.real * 10**decimals) / 10**decimals + 1j * (np.trunc(operator.imag * 10**decimals) / 10**decimals)

def real_diag(operator):
    """
    Deletes the imaginary part of an operator diagonal.

    Parameters
    ----------
    operator : Numpy array of an operator

    Returns
    -------
    A Numpy array with real diagonal values

    """

    np.fill_diagonal(operator, np.diag(operator).real)

def negativity(operator):
    """
    Calculates the negativity of an operator.

    Parameters
    ----------
    operator : Numpy array of an operator

    Returns
    -------
    A float value of the negativity value of the operator

    """

    return (trace_norm(operator)-1)/2

def logarithmic_negativity(operator):
    """
    Calculates the logarithmic negativity of an operator.

    Parameters
    ----------
    operator : Numpy array of an operator

    Returns
    -------
    A float logarithmic negativity value of the operator

    """

    return np.log2(trace_norm(operator))

def assess_dm_entanglement(dm, part, dims_first, dims_second, rounding=None):
    """
    Determines the entanglement of a two-mode density matrix.

    Parameters
    ----------
    dm : Numpy array of the combined density matrix
    part : String value of the partially transposed system
    dims_first : Dimensions of the first system
    dims_second : Dimension of the second system
    rounding : Int value of the amount of decimals kept

    Returns
    -------
    A float entanglement value of the system

    """

    if rounding:
        return truncate_mantissa(logarithmic_negativity(partial_transpose(dm, part, dims_first, dims_second)),rounding)
    return logarithmic_negativity(partial_transpose(dm, part, dims_first, dims_second))

def get_entanglement_values(dms, part, dims_first, dims_second, rounding=None):
    """
    Determines the entanglement of a set of two-mode density matrices.

    Parameters
    ----------
    dms : A set of Numpy arrays of combined density matrices
    part : String value of the partially transposed system
    dims_first : Dimensions of the first system
    dims_second : Dimension of the second system
    rounding : Int value of the amount of decimals kept

    Returns
    -------
    A Numpy array of entanglement values of the system. 
    If the system is entangled the entry is [1,0] and [0,1] otherwise.

    """

    return np.array([[1,0] if assess_dm_entanglement(dm, "first", dims_first, dims_second, rounding) > 0 else [0,1] for dm in dms])

def expectation_value(dm, operator):
    """
    Calculates the expectation value of an operator.

    Parameters
    ----------
    dm : Numpy array of the density matrix of a system
    operator : Numpy array of an operator

    Returns
    -------
    The measured expectation value

    """

    return np.trace(dm @ operator)

def rk4(function, initial_state, t, dt):
    """
    Calculates one step of the 4th order Runge-Kutta numerical simulation for some IVP y'=f(t,y)

    Parameters
    ----------
    function : The function which is solved numerically with RK4
    initial_state : Initial state of the IVP
    t : Current time of IVP
    dt : Time-step

    Returns
    -------
    The value of the function after one step of RK4

    """

    k1_ = function(initial_state, t)
    k2_ = function(initial_state + 0.5 * dt * k1_, t + 0.5 * dt)
    k3_ = function(initial_state + 0.5 * dt * k2_, t + 0.5 * dt)
    k4_ = function(initial_state + dt * k3_, t + dt)
    return initial_state + (dt/6)*(k1_ + 2 * k2_ + 2 * k3_ + k4_)

def data_standardize(data):
    """
    Standardizes data array values to 0 mean and 1 std.

    Parameters
    ----------
    data : Numpy array of the data to be standardized

    Returns
    -------
    A standardized data array

    """

    return (data - np.mean(data, axis=0))/np.std(data, axis=0)

def init_identity(size):
    """
    Initializes a identity operator.

    Parameters
    ----------
    size : Int value of the dimensions of the identity operator

    Returns
    -------
    A Numpy array of an identity operator

    """

    return np.eye(size)

def init_destroy(size):
    """
    Initializes a destruction field operator.

    Parameters
    ----------
    size : Int value of the degrees of freedom of the operator

    Returns
    -------
    A Numpy array of a destruction operator

    """

    return np.diag(np.sqrt(np.arange(1,size)),1)

def init_create(size):
    """
    Initializes a creation field operator.

    Parameters
    ----------
    size : Int value of the degrees of freedom of the operator

    Returns
    -------
    A Numpy array of a creation operator

    """

    return np.diag(np.sqrt(np.arange(1,size)),-1)

def init_dissipator(dm, operator1, operator2=None):
    """
    Initializes a Lindblad dissipator.

    Parameters
    ----------
    dm : Numpy array of the density matrix of a system
    operator1 : Numpy array of the operator
    operator2 : Numpy array of the operator

    Returns
    -------
    A Numpy array of the Lindblad dissipator

    """

    if not operator2:
        operator2 = operator1

    return operator1 @ dm @ dagger(operator2) - 0.5 * dagger(operator1) @ operator2 @ dm - 0.5 * dm @ dagger(operator1) @ operator2 

def init_dissipators(dm, operators):
    """
    Initializes a set of Lindblad dissipators.

    Parameters
    ----------
    dm : Numpy array of the density matrix of a system
    operators : Set of Numpy arrays of operators

    Returns
    -------
    A sum of Numpy arrays of the Lindblad dissipators collected into one array

    """

    return sum([init_dissipator(dm, op) for op in operators])

def init_multipartite_dc_operators(doperators, sizes):
    """
    Initializes a set of destruction and creation field operators for a composite system.

    Parameters
    ----------
    doperators : Set of Numpy arrays of destruction operators of each system
    sizes : Int of the amount of subsystems in each system
    
    Returns
    -------
    doperators_ : Numpy arrays of destruction operators of each subsystem in each system
    coperators_ : Numpy arrays of creation operators of each subsystem in each system  

    """

    identities_ = [init_identity(x.shape[0]) for x in doperators]
    part_identities_ = [init_identity(x.shape[0] ** y) for x,y in zip(doperators, sizes)]

    doperators_ = [[tensor([*part_identities_[:idx], *x, *part_identities_[idx+1:]]) for x in \
                    [[dop if i==j else nochange for j, nochange in enumerate(site)] for i, site in enumerate([id for _ in range(size)] for _ in range(size))]] for idx, (dop, id, size) in enumerate(zip(doperators, identities_, sizes))]
    coperators_ = [[tensor([*part_identities_[:idx], *x, *part_identities_[idx+1:]]) for x in \
                    [[dagger(dop) if i==j else nochange for j, nochange in enumerate(site)] for i, site in enumerate([id for _ in range(size)] for _ in range(size))]] for idx, (dop, id, size) in enumerate(zip(doperators, identities_, sizes))]

    return doperators_, coperators_

def init_sq(alpha, truncate):
    """
    Initializes a single mode squeezing operator

    Parameters
    ----------
    alpha : Complex value of the squeezing parameter
    truncate : Int value of the degrees of freedom

    Returns
    -------
    A Numpy array of the single mode squeezing operator

    """

    a_ = init_destroy(truncate)
    A_ = 0.5 * (alpha * dagger(a_) @ dagger(a_) - np.conjugate(alpha) * a_ @ a_)
    return sp.linalg.expm(A_)

def init_two_mode_sq(alpha, a1, a2):
    """
    Initializes a two-mode squeezing operator

    Parameters
    ----------
    alpha : Complex value of the squeezing parameter
    a1 : Numpy array of the first mode destruction operator
    a2 : Numpy array of the second mode destruction operator

    Returns
    -------
    A Numpy array of the two-mode squeezing operator

    """
    
    A_ = alpha * dagger(a1) @ dagger(a2) - np.conjugate(alpha) * a1 @ a2
    return sp.linalg.expm(A_)

def init_th(mean_n, truncate, small_trunc_norm=False):
    """
    Initializes a single mode mode thermal state.

    Parameters
    ----------
    mean_n : Float value of the average thermal number
    truncate : Int value of the degrees of freedom
    small_trunc_norm : Boolean value to use a different normalization for values

    Returns
    -------
    A Numpy array of the single mode thermal state density matrix

    """
    
    if small_trunc_norm:
        beta = np.log(1.0 / mean_n + 1.0)
        values_ = np.exp(-beta * np.arange(truncate))
        values_ = values_ / np.sum(values_)
    else:
        values_ = [1/(1+mean_n)*(mean_n/(1+mean_n))**x for x in np.arange(truncate)]

    return np.diag(values_)

def init_two_mode_th(mean_n, truncate):
    """
    Initializes a two-mode thermal state.

    Parameters
    ----------
    mean_n : Float value of the average thermal number
    truncate : Int value of the degrees of freedom

    Returns
    -------
    A Numpy array of the two-mode thermal state density matrix

    """
    
    return tensor([init_th(mean_n, truncate, other=True), init_th(mean_n, truncate,other=True)])

def init_vac(truncate):
    """
    Initializes a vacuum state.

    Parameters
    ----------
    truncate : Int value of the degrees of freedom

    Returns
    -------
    vac_ : A Numpy array of the vacuum state density matrix

    """
    
    vac_ = np.zeros((truncate, truncate), dtype=np.complex64)
    vac_[0,0] = 1
    return vac_

def init_ST(alpha, mean_n, truncate, a1, a2, rounding=None):
    """
    Initializes a two-mode squeezed thermal state.

    Parameters
    ----------
    alpha : Complex value of the squeezing parameter
    mean_n : Float value of the average thermal number
    truncate : Int value of the degrees of freedom
    a1 : Numpy array of the first mode destruction operator
    a2 : Numpy array of the second mode destruction operator
    rounding : Int value of the amount of decimals kept

    Returns
    -------
    A Numpy array of the ST state density matrix

    """

    sq_ = init_two_mode_sq(alpha, a1, a2)
    th_ = init_two_mode_th(mean_n, truncate)

    if rounding:  
        return truncate_mantissa(sq_ @ th_ @ dagger(sq_), rounding)
    
    return sq_ @ th_ @ dagger(sq_)

def init_PASV(alpha, truncate, a1, a2, rounding=None):
    """
    Initializes an entangled two-mode photon added squeezed vacuum state.

    Parameters
    ----------
    alpha : Complex value of the squeezing parameter
    truncate : Int value of the degrees of freedom
    a1 : Numpy array of the first mode destruction operator
    a2 : Numpy array of the second mode destruction operator
    rounding : Int value of the amount of decimals kept

    Returns
    -------
    sq_add_ : A Numpy array of an entangled PASV state density matrix

    """
    
    sq_ = init_two_mode_sq(alpha, a1, a2)
    vac_ = init_vac(truncate**2)
    sq_add_ = dagger(a1) @ dagger(a2) @ sq_ @ vac_ @ dagger(sq_) @ a2 @ a1

    #Normalize
    sq_add_ = sq_add_ / np.trace(sq_add_)

    if rounding:  
        return truncate_mantissa(sq_add_, rounding)
    
    return sq_add_

def init_PASV_sep(alpha1, alpha2, truncate, a, rounding=None):
    """
    Initializes a separable two-mode photon added squeezed vacuum state.

    Parameters
    ----------
    alpha1 : Complex value of the first squeezing parameter
    alpha2 : Complex value of the second squeezing parameter
    truncate : Int value of the degrees of freedom
    a : Numpy array of the destruction operator
    rounding : Int value of the amount of decimals kept

    Returns
    -------
    A Numpy array of a separable PASV state density matrix

    """

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
    """
    Initializes an entangled two-mode photon subtracted squeezed vacuum state.

    Parameters
    ----------
    alpha : Complex value of the squeezing parameter
    truncate : Int value of the degrees of freedom
    a1 : Numpy array of the first mode destruction operator
    a2 : Numpy array of the second mode destruction operator
    rounding : Int value of the amount of decimals kept

    Returns
    -------
    sq_sub_ : A Numpy array of an entangled PSSV state density matrix

    """

    sq_ = init_two_mode_sq(alpha, a1, a2)
    vac_ = init_vac(truncate**2)
    sq_sub_ = a1 @ a2 @ sq_ @ vac_ @ dagger(sq_) @ dagger(a2) @ dagger(a1)

    #Normalize
    sq_sub_ = sq_sub_ / np.trace(sq_sub_)

    if rounding:  
        return truncate_mantissa(sq_sub_, rounding)
    
    return sq_sub_

def init_PSSV_sep(alpha1, alpha2, truncate, a, rounding=None):
    """
    Initializes a separable two-mode photon subtracted squeezed vacuum state.

    Parameters
    ----------
    alpha1 : Complex value of the first squeezing parameter
    alpha2 : Complex value of the second squeezing parameter
    truncate : Int value of the degrees of freedom
    a : Numpy array of the destruction operator
    rounding : Int value of the amount of decimals kept

    Returns
    -------
    A Numpy array of a separable PSSV state density matrix

    """

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
    """
    Initializes an entangled two-mode number state c0|00>+c1|11>.

    Parameters
    ----------
    alpha : Complex value of the squeezing parameter
    truncate : Int value of the degrees of freedom
    a1 : Numpy array of the first mode destruction operator
    a2 : Numpy array of the second mode destruction operator
    rounding : Int value of the amount of decimals kept

    Returns
    -------
    sq_add_ : A Numpy array of an entangled NUM state density matrix

    """

    vac_site_ = np.zeros((truncate**2,), dtype=np.complex64)
    vac_site_[0] = c0
    exit_site_ = np.zeros((truncate**2,), dtype=np.complex64)
    exit_site_[truncate+1] = c1
    NUM_ = np.outer(vac_site_ + exit_site_, np.conj(vac_site_ + exit_site_))

    if rounding:
        return truncate_mantissa(NUM_, rounding)
    
    return NUM_

def init_NUM_sep(truncate, rounding=None):
    """
    Initializes a separable two-mode number state |00> or |11>.

    Parameters
    ----------
    truncate : Int value of the degrees of freedom
    rounding : Int value of the amount of decimals kept

    Returns
    -------
    A Numpy array of a separable NUM state density matrix

    """

    NUM_ = np.zeros((truncate**2,truncate**2), dtype=np.complex64)
    if np.random.choice(2,1):
        NUM_[0,0] = 1
    else:
        NUM_[truncate+1,truncate+1] = 1

    if rounding:
        return truncate_mantissa(NUM_, rounding)
    
    return NUM_

def gen_input_states(type, amount_of_states, truncate, entanglement=False, rounding=None):
    """
    Generates two-mode input states in bulk.

    Parameters
    ----------
    type : String value of the generated input types
    amount_of_states : Int value of the amount of states generated
    truncate : Int value of the degrees of freedom
    entanglement : Boolean value indicating if the given states' entanglement values should be provided
    rounding : Int value of the amount of decimals kept
    
    Returns
    -------
    A tuple of a set of Numpy arrays of input states and their entanglement values

    """

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
    
    #Entangled two-mode photon added squeezed vacuum states
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

    #Separable two-mode photon added squeezed vacuum states
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

    #50/50 split of separable and entangled two-mode photon added squeezed vacuum states
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

    #Entangled two-mode photon subtracted squeezed vacuum states
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

    #Separable two-mode photon subtracted squeezed vacuum states
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

    #50/50 split of separable and entangled two-mode photon added squeezed vacuum states
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

    #Entangled two-mode number states  
    elif type == "NUM":
        theta_NUM_ = np.array([0.5*np.arcsin(x) for x in np.random.uniform(0,1,(amount_of_states,))])
        phi_NUM_ = np.random.uniform(0,2*np.pi,(amount_of_states,))
        c0_NUM_ = np.array([np.sin(x) for x in theta_NUM_])
        c1_NUM_ = np.array([np.cos(x)*np.exp(1j*y) for x,y in zip(theta_NUM_, phi_NUM_)])

        NUM_ = np.array([init_NUM(x,y,truncate,rounding) for x,y in zip(c0_NUM_, c1_NUM_)])

        if entanglement:
            return NUM_, np.array([[1,0] for _ in range(amount_of_states)])
        return NUM_  

    #Separable two-mode number states    
    elif type == "NUM_sep":

        NUM_sep_ = np.array([init_NUM_sep(truncate,rounding) for _ in range(amount_of_states)])

        if entanglement:
            return NUM_sep_, np.array([[0,1] for _ in range(amount_of_states)])
        return NUM_sep_  

    #50/50 split of separable and entangled two-mode number states   
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
    """
    Unpacks the relevant parameters used in the creation and simulation of the Qreservoir class from a .json file.

    Parameters
    ----------
    filepath : String value of the .json file filepath

    Returns
    -------
    A tuple of the relevant values for a Qreservoir class simulation

    """

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

def gen_narma_nmse_filepath(narma_degree):
    """
    Generates a filepath for writing results of NARMA evaluated with NMSE.
    Assumes a directory of the variable 'dictname' exists in the working directory.

    Parameters
    ----------
    narma_degree : Int value of the degree of NARMA

    Returns
    -------
    A string value of the filepath
    
    """

    dictname = "reservoir_narma_nmse"
    filename = "narma" + str(narma_degree) + "_nmse.csv"
    return os.path.join(os.getcwd(),dictname, filename)

def gen_config_filepath(confignumber):
    """
    Generates a filepath to fetch a given configuration of Qreservoir class simulation.
    Assumes a directory of the variable 'dictname' exists in the working directory.

    Parameters
    ----------
    confignumber : Int value of the number of the configuration

    Returns
    -------
    A string value of the filepath
    
    """

    dictname = "Qreservoir_configurations"
    filename = "config" + str(confignumber) + ".json"
    return os.path.join(os.getcwd(),dictname, filename)

def system_name(gamma, res_size, fock_trunc, res_connect, cm=False):
    """
    Generates a system name for writing the entanglement classification results of Qreservoir class.

    Parameters
    ----------
    gamma : Float value used in the Qreservoir class simulations
    res_size : Int value of the reservoir size used in the Qreservoir class simulations
    fock_trunc : Int value of the input degrees of freedom used in the Qreservoir class simulations
    res_connect : String value of reservoir connectivity used in the Qreservoir class simulation
    cm : Boolean value indicating if the results will be written for a confusion matrix

    Returns
    -------
    A string value of the system name
    
    """

    if cm:
        return str(gamma) + "_" + str(res_size) + "_" + str(fock_trunc) + "_" + str(res_connect) + "_cm"
    return str(gamma) + "_" + str(res_size) + "_" + str(fock_trunc) + "_" + str(res_connect)

def gen_system_filepath(folder, systemnumber):
    """
    Generates a filepath for saving the pretrained Qreservoir systems with all relevant parameters.
    Assumes a directory of the variable 'dictname' exists in the working directory.

    Parameters
    ----------
    folder : String value of the folder name generated by function 'system_name'
    systemnumber : Int value of the number of the given system in an ensemble of systems

    Returns
    -------
    A string value of the filepath
    
    """

    dictname = "Qreservoir_systems"
    filename = "system" + str(systemnumber) + ".npz"
    return os.path.join(os.getcwd(),dictname, folder, filename)

def gen_result_filepath(filename):
    """
    Generates a filepath for writing entanglement classification results of Qreservoir class.
    Assumes a directory of the variable 'dictname' exists in the working directory.

    Parameters
    ----------
    filename : String value of the system name generated by function 'system_name'

    Returns
    -------
    A string value of the filepath
    
    """

    dictname = "Qreservoir_results"
    filename = filename + ".csv"
    return os.path.join(os.getcwd(),dictname, filename)

def write_to_row(value, filepath):
    """
    Writes a string value + ';' to a file

    Parameters
    ----------
    value : String value to be written in the file
    filepath : String value of the filepath

    """

    with open(filepath, 'a') as file:
        file.write(value + ";")

def finish_row(filepath):
    """
    Finishes a row in a file.

    Parameters
    ----------
    filepath : String value of the filepath
    
    """
    with open(filepath, 'a') as file:
        file.write('0\n')