

#Where good ideas go to die




#Evolves system for one input pair
def time_evolve(cov_M_: np.array, A_: np.array, time_step_: float, tau_: float):
    #Initialize first mode
    first_mode = copy.copy(A_)
    first_mode[:,[2,3]] = 0

    eigenvalues_first_, X1_ = np.linalg.eig(first_mode)
    diagonal_eig1_ = np.diag(eigenvalues_first_)
    lambda1_M_ = np.kron(diagonal_eig1_, np.eye(12)) + np.kron(np.eye(12), diagonal_eig1_)
    Q1_ = np.kron(X1_, X1_)

    #Initialize second mode
    second_mode = copy.copy(A_)
    second_mode[:,[0,1]] = 0

    eigenvalues_second_, X2_ = np.linalg.eig(second_mode)
    diagonal_eig2_ = np.diag(eigenvalues_second_)
    lambda2_M_ = np.kron(diagonal_eig2_, np.eye(12)) + np.kron(np.eye(12), diagonal_eig2_)
    Q2_ = np.kron(X2_, X2_)

    #Vectorize initial state
    v_t_ = cov_M_.flatten('F')

    #First mode interaction
    for i in np.arange(time_step_, tau_+time_step_, time_step_):
        v_t_ = Q1_ @ sp.linalg.expm(lambda1_M_*time_step_) @ np.linalg.inv(Q1_) @ v_t_
        #print(np.round(v_t_.reshape((12,12),order='F').real.diagonal(),3))

    #Second mode interaction
    for i in np.arange(tau_+time_step_, 2*tau_+time_step_, time_step_):
        v_t_ = Q2_ @ sp.linalg.expm(lambda2_M_*time_step_) @ np.linalg.inv(Q2_) @ v_t_
        #print(np.round(v_t_.reshape((12,12),order='F').real.diagonal(),3))

    #Return the time evolved covariance matrix    
    return v_t_.reshape((12,12),order='F').real

#Evolves system for one input pair
def time_evolve(cov_M_: np.array, A_: np.array, tau_: float):

    interaction_time_ = (0, tau_)

    #Initialize first mode interaction matrix
    first_mode_ = copy.copy(A_)
    first_mode_[:,[2,3]] = 0
    #Initialize second mode interaction matrix
    second_mode_ = copy.copy(A_)
    second_mode_[:,[0,1]] = 0

    def lyapunov_first(t_, vec_cov_M_):
        cov_M_ = vec_cov_M_.reshape((12, 12))
        return (first_mode_ @ cov_M_ + cov_M_ @ first_mode_.T).flatten()

    def lyapunov_second(t_, vec_cov_M_):
        cov_M_ = vec_cov_M_.reshape((12, 12))
        return (second_mode_ @ cov_M_ + cov_M_ @ second_mode_.T).flatten()

    #Simulate interaction for both modes
    first_interact = sp.integrate.solve_ivp(fun = lyapunov_first, t_span = interaction_time_, y0 = cov_M_.flatten(), method = "RK45")
    second_interact = sp.integrate.solve_ivp(fun = lyapunov_second, t_span = interaction_time_, y0 = first_interact.y[:,-1], method = "RK45")

    #Return the time-evolved covariance matrix
    return second_interact.y[:,-1].reshape(12,12)

#Evolves system for one input pair
def time_evolve1(cov_M_: np.array, first_mode_: np.array, second_mode_:np.array, nointeract_mode_: np.array, time_steps_: np.float64, tau_: np.float64):

    def piecewice(t_, _vec_cov_M_):
        t_ = np.round(t_, 6)
        if 0 < t_ < tau_:
            return first_mode_ @ _vec_cov_M_
        elif tau_ < t_ < 2*tau_:
            return second_mode_ @ _vec_cov_M_
        else:
            return nointeract_mode_ @ _vec_cov_M_
    
    step_size_ = np.round(2*tau_/time_steps_,6)

    vec_cov_M_ = cov_M_.flatten()

    def rk4(function_, t_, _vec_cov_M_, step_):
        k1_ = function_(t_, _vec_cov_M_)
        k2_ = function_(t_ + 0.5 * step_, _vec_cov_M_ + 0.5 * step_ * k1_)
        k3_ = function_(t_ + 0.5 * step_, _vec_cov_M_ + 0.5 * step_ * k2_)
        k4_ = function_(t_ + step_, _vec_cov_M_ + step_ * k3_)
        return _vec_cov_M_ + (step_/6)*(k1_ + 2 * k2_ + 2 * k3_ + k4_)

    print(np.round(vec_cov_M_, 2))
    for current_time in np.arange(0, 2*tau_, step_size_):
        vec_cov_M_ = rk4(piecewice, current_time, vec_cov_M_, step_size_)
        print(np.round(vec_cov_M_, 2))
    
    #Return the time-evolved covariance matrix
    return vec_cov_M_.reshape(12,12)



#Initialize covariance matrix of system
cov_full = init_cov_full()
first_mode, second_mode, nointeract_mode, matrix_A = init_A(gamma, P, W_in, J_ij, eta)

time_steps = 8
W_out = np.random.uniform(0,1,(reservoir_size,2))
Y_out_pred = np.empty(number_of_inputs, dtype=tuple)
for idx, input in enumerate(input_states_train):
    inject_input(cov_full, input)
    print(idx)
    cov_full = time_evolve(cov_full, first_mode, second_mode, nointeract_mode, time_steps, tau)
    occupation_nums_ = get_occupation_nums(cov_full, reservoir_size)
    Y_true_ = Y_out_true[idx]
    print(np.round(cov_full,2))
    #print(Y_true_)
    #print(predict_Y_out(occupation_nums_, W_out))
    Y_pred_ = Y_out_pred[idx] = predict_Y_out(occupation_nums_, W_out)
    update_W_out(learning_rate, regularization_strength, Y_true_, Y_pred_, occupation_nums_, W_out, reservoir_size)
    #print(occupation_nums_)
    print(W_out)
    print("")


#Updates the output weights of network
def update_W_out(learning_rate_: np.float64, reg_strength_: np.float64, Y_true_: tuple, Y_pred_: tuple, occupation_numbers_: np.array, W_out_: np.array, reservoir_size_: np.float64):
    for i in range(reservoir_size_):
        W_out_[i] = W_out_[i] - learning_rate_*(-occupation_numbers_[i]*(Y_true_ - Y_pred_) + reg_strength_*W_out_[i])


#Function that creates a two-mode squeezed thermal state
def init_sq_th(alpha_: np.float64, mean_n_: np.float64, truncate: int, a1_, a2_):

    #Initialize the tmo-mode squeezing matrix and two-mode thermal state
    two_mode_sq_ = qt.squeezing(a1_, a2_, -2*alpha_)
    two_mode_th_ = qt.tensor(qt.thermal_dm(truncate, mean_n_), qt.thermal_dm(truncate, mean_n_))

    #Return two-mode squeezed thermal state
    return two_mode_sq_ @ two_mode_th_ @ two_mode_sq_.dag()

def init_pho_add(alpha_: np.float64, truncate: int, a1_, a2_):
    
    #Initialize two-mode squeezing matrix and vacuum state 
    two_mode_sq_ = qt.squeezing(a1_, a2_, 2*alpha_)
    two_mode_vac_ = qt.tensor(qt.states.fock_dm(truncate,0), qt.states.fock_dm(truncate,0))

    #Return two-mode photon added squeezed state
    return  a1_.dag() @ a2_.dag() @ two_mode_sq_ @ two_mode_vac_ @ two_mode_sq_.dag() @ a2_ @ a1_

def init_pho_sub(alpha_: np.float64, truncate: int, a1_, a2_):

    #Initialize two-mode squeezing matrix and vacuum state 
    two_mode_sq_ = qt.squeezing(a1_, a2_, 2*alpha_)
    two_mode_vac_ = qt.tensor(qt.states.fock_dm(truncate,0), qt.states.fock_dm(truncate,0))

    #Return two-mode photon subtracted squeezed state
    return  a1_ @ a2_ @ two_mode_sq_ @ two_mode_vac_ @ two_mode_sq_.dag() @ a2_.dag() @ a1_.dag()

def init_simple(c0_: np.float64, c1_: np.float64, truncate: int):

    #Create the state according to the paper
    state_ = c0_*qt.tensor(qt.states.basis(truncate,0), qt.states.basis(truncate,0)) + c1_*qt.tensor(qt.states.basis(truncate,1), qt.states.basis(truncate,1))

    return state_ @ state_.dag()