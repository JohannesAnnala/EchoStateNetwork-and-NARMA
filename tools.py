import numpy as np
import copy

def init_identity(size):
    return np.eye(size)

def dagger(operator):
    return np.conjugate(operator).T

def tensor(operators):
    product_ = operators[0]
    for i in range(1,len(operators)):
        product_ = np.kron(product_, operators[i])

    return product_

#Mathematics for partial trace in "Computing partial traces and reduced density matrices" by Jonas Maziero
def partial_trace(rho, part, new_dims):
    rho_new_ = np.zeros((new_dims,new_dims))
    
    if part == "first":
        for row in range(new_dims):
            for column in range(new_dims):
                rho_new_[row,column] = np.trace(rho[new_dims*row:new_dims*(row+1), new_dims*column:new_dims*(column+1)])
    elif part == "second":
        for block in range(new_dims):
            rho_new_ += rho[new_dims*block:new_dims*(block+1), new_dims*block:new_dims*(block+1)]
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

def negativity(rho):
    eig_vals_ = np.linalg.eigvals(rho)
    return np.sum((np.absolute(eig_vals_)-eig_vals_)/2)

def rk4(function, initial_state, t, dt):
    k1_ = function(initial_state, t)
    k2_ = function(initial_state + 0.5 * dt * k1_, t + 0.5 * dt)
    k3_ = function(initial_state + 0.5 * dt * k2_, t + 0.5 * dt)
    k4_ = function(initial_state + dt * k3_, t + dt)
    return initial_state + (dt/6)*(k1_ + 2 * k2_ + 2 * k3_ + k4_)
