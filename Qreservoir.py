import numpy as np
from itertools import combinations
from tools import init_identity, tensor, init_destroy, init_multipartite_dc_operators, init_vac, truncate_mantissa, data_standardize, partial_trace, expectation_value

from sklearn.linear_model import Ridge
import copy
import warnings

warnings.simplefilter("always", RuntimeWarning)

class QReservoir:
    """
    A class for entanglement classification of Gaussian and non-Gaussian states.
    The classification is done with quantum reservoir computing on a fermionic reservoir.

    Attributes
    ----------
    reservoir_size : The amount of modes in the reservoir
    energy_truncate_level : The input mode degrees of freedom
    dims : A tuple of the dimensions of input and reservoir density matrices
    size : The dimensions of the combined density matrix
    reservoir_connectivity : The connectivity of the reservoir network
    sim_precision : The amount of decimals the simulation keeps as it's precision
    gamma : Float value of the dissipation coefficient
    P : Float value of the pumping strength
    W_in : Numpy array of the input weights
    eta : Float value of the eta coefficient in simualtion
    tau : Float value of one mode interaction time with the reservoir
    a : Numpy array of the destruction operator of an input mode
    a_id : Numpy array of the identity operator of an input mode
    b : Numpy array of the destruction operator of a reservoir mode
    b_id : Numpy array of the identity operator of a reservoir mode
    a_system : A set of Numpy arrays of all input mode destruction operators
    a_dag_system : A set of Numpy arrays of all input mode creation operators
    b_system : A set of Numpy arrays of all reservoir mode destruction operators
    b_dag_system : A set of Numpy arrays of all reservoir mode destruction operators
    H_unitary : Numpy array of the unitary Hamiltonian dynamics
    A : Numpy array of an operator of dynamics
    B : Numpy array of an operator of dynamics
    C : Numpy array of an operator of dynamics
    D : Numpy array of an operator of dynamics
    b_system_gamma : A set of Numpy arrays of all reservoir mode destruction operators multiplied with 'gamma'
    b_dag_system_P : A set of Numpy arrays of all reservoir mode creation operators multiplied with 'P'
    b_dag_b : A set of Numpy arrays of all 'b^(dagger)b'
    Ridge_param : The regularization strength used in ridge regression
    entangled_forecast : Scikit learn 'Ridge' model for entanglement prediction
    separable_forecast : Scikit learn 'Ridge' model for separable prediction
    step : The step size for RK4
    timesteps : Numpy array of all steps
    rho_full : The combined density matrix
    train_measured_observable : Numpy array of the measured average occupation numbers during training
    train_Y_true : Numpy array of the true entanglement values of training inputs
    rho_full_after_train : The state of the reservoir after the training
    test_measured_observables : Numpy array of the measured average occupation numbers during testing
    test_Y_true : Numpy array of the true entanglement values of testing inputs
    test_prob_Y_pred : Numpy array of the predicted probabilities of the entanglement values of testing inputs
    test_Y_pred : Numpy array of the predicted entanglement values of testing inputs

    Methods
    -------
    change_reservoir_constant() : Change constant reservoir values
    init_reservoir() : Initializes the reservoir to an initial state before training
    init_unitary_evolution() : Initializes the unitary dynamics of the reservoir
    rk4_timesteps() : Sets the rounds of RK4 calculated for each input
    inject_input() : Changes the input state interacting with the reservoir
    update_reservoir() : Simulates the reservoir for one input
    measure_reservoir() : Measures the reservoir average occupation numbers
    update_and_measure_reservoir() : Evolves the reservoir for a set of inputs and collects the average occupation numbers after each interaction
    assign_entanglement_from_probabilities() : Assigns entanglement values of inputs based on predictions
    analyze_performance() : Analyzes the entanglement classification performance
    train_reservoir() : Trains the quantum reservoir computing system
    test_reservoir() : Tests the quantum reservoir computing system
    reset_after_test() : Reset the state of the reservoir to the state after training
    system_save() : Saves a trained QRC system to a .npz file for later use
    system_load() : Loads a trained QRC sustem from a .npz file for testing
    init_from_file() : A class method to initialize a new QReservoir object from a .npz file

    """

    def __init__(self, gamma=1, reservoir_size=4, energy_truncate_level=5, reservoir_connectivity="alltoall", sim_precision=12):
        self.reservoir_size = reservoir_size
        self.energy_truncate_level = energy_truncate_level
        self.dims = (energy_truncate_level**2, 2**reservoir_size)
        self.size = (energy_truncate_level**2)*(2**reservoir_size)
        if (reservoir_connectivity == "alltoall") | (reservoir_connectivity == "ring") | (reservoir_connectivity == "sausage"):
            self.reservoir_connectivity = reservoir_connectivity
        else:
            raise ValueError("Invalid value for reservoir connectivity, try 'alltoall', 'ring' or 'sausage'")
        self.sim_precision = sim_precision
        self.gamma = gamma
        self.P = 0.1*gamma

        self.W_in = np.random.uniform(0,gamma,(reservoir_size,))
        self.eta = sum(self.W_in**2)
        self.tau = 1/gamma

        self.a = init_destroy(energy_truncate_level)
        self.a_id = init_identity(energy_truncate_level)
        self.b = init_destroy(2)
        self.b_id = init_identity(2)

        system_dops_, system_cops_ = init_multipartite_dc_operators([self.a,self.b],[2,reservoir_size])
        self.a_system, self.a_dag_system, self.b_system, self.b_dag_system = system_dops_[0], system_cops_[0], system_dops_[1], system_cops_[1]
        self.init_unitary_evolution()

        self.A = np.array([-1j * self.H_unitary + sum([- 0.5 * gamma * b_dag @ b - 0.5 * self.P * b @ b_dag for b, b_dag in zip(self.b_system,self.b_dag_system)]),
                           *[-1j * self.H_unitary + sum([- 0.5 * gamma * b_dag @ b - 0.5 * self.P * b @ b_dag - w_in * b_dag @ a for b, b_dag, w_in in zip(self.b_system,self.b_dag_system,self.W_in)]) - \
                             0.5 * (self.eta/gamma) * a_dag @ a for a, a_dag in zip(self.a_system,self.a_dag_system)]])

        self.B = np.array([1j * self.H_unitary + sum([- 0.5 * gamma * b_dag @ b - 0.5 * self.P * b @ b_dag for b, b_dag in zip(self.b_system,self.b_dag_system)]),
                           *[1j * self.H_unitary + sum([- 0.5 * gamma * b_dag @ b - 0.5 * self.P * b @ b_dag - w_in * a_dag @ b for b, b_dag, w_in in zip(self.b_system,self.b_dag_system,self.W_in)]) - \
                             0.5 * (self.eta/gamma) * a_dag @ a for a, a_dag in zip(self.a_system,self.a_dag_system)]])

        self.C = np.array([[gamma * b + w_in * a for b, w_in in zip(self.b_system,self.W_in)] for a in self.a_system])

        self.D = np.array([sum([w_in * b for b, w_in in zip(self.b_system,self.W_in)]) + (self.eta/gamma) * a for a in self.a_system])

        self.b_system_gamma = np.array([gamma * b for b in self.b_system])
        self.b_dag_system_P = np.array([0.1 * gamma * b_dag for b_dag in self.b_dag_system])

        self.b_dag_b = np.array([b_dag @ b for b, b_dag in zip(self.b_system,self.b_dag_system)])

        self.Ridge_param = 0.1
        self.entangled_forecast = Ridge(self.Ridge_param)
        self.separable_forecast = Ridge(self.Ridge_param)

    def change_reservoir_constants(self, gamma=None, truncate=None, reservoir_size=None, reservoir_connectivity=None):
        """
        Changes the constant values regarding the reservoir.

        Parameters
        ----------
        gamma : Float value of the dissipation coefficient
        truncate : Int value of the input mode degrees of freedom
        reservoir_size : Int value of the amount of reservoir modes
        reservoir_connectivity : String value of the reservoir connectivity

        """

        if gamma:
            self.gamma = gamma
            self.P = 0.1*gamma
            self.W_in = np.random.uniform(0,gamma,(self.reservoir_size,))
            self.eta = sum(self.W_in**2)
            self.tau = 1/gamma
        if truncate:
            self.energy_truncate_level = truncate
            self.a = init_destroy(truncate)
            self.a_id = init_identity(truncate)
            new_dops_, new_cops_ = init_multipartite_dc_operators([self.a, self.b], [2, self.reservoir_size])
            self.a_system, self.b_system, self.b_dag_system = new_dops_[0], new_dops_[1], new_cops_[1] 
        if reservoir_size:
            self.reservoir_size = reservoir_size      
            new_dops_, new_cops_ = init_multipartite_dc_operators([self.a, self.b], [2, self.reservoir_size])
            self.a_system, self.b_system, self.b_dag_system = new_dops_[0], new_dops_[1], new_cops_[1] 
        if reservoir_connectivity:
            self.reservoir_connectivity = reservoir_connectivity
            self.init_unitary_evolution()            

    def init_reservoir(self, initial_state):
        """
        Initializes the reservoir before training.

        Parameters
        ----------
        initial_state : String value of the initial state chosen

        """

        if initial_state == "vacuum":

            self.rho_full = init_vac(self.dims[0] * self.dims[1])
        else:
            raise ValueError("Quantum reservoir can't be initialized in such an initial state, try 'vacuum'")

    def init_unitary_evolution(self):
        """
        Initializes the unitary dynamics of the simulation.

        """

        dims_ = self.dims[0] * self.dims[1]
        H_unitary_ = np.zeros((dims_, dims_))

        if self.reservoir_connectivity == "alltoall":
            J_ij_ = np.random.uniform(-self.gamma, self.gamma, int((self.reservoir_size*(self.reservoir_size-1)/2),))

            for (first, second), interaction_str in zip(combinations(range(self.reservoir_size), 2), J_ij_):
                H_unitary_ += interaction_str*(self.b_dag_system[first] @ self.b_system[second] + self.b_dag_system[second] @ self.b_system[first])

        elif self.reservoir_connectivity == "ring":
            J_ij_ = np.random.uniform(-self.gamma, self.gamma, (self.reservoir_size,))

            for i in range(self.reservoir_size-1):
                H_unitary_ += J_ij_[i]*(self.b_dag_system[i] @ self.b_system[i+1] + self.b_dag_system[i+1] @ self.b_system[i])

            H_unitary_ += J_ij_[self.reservoir_size-1]*(self.b_dag_system[self.reservoir_size-1] @ self.b_system[0] + self.b_dag_system[0] @ self.b_system[self.reservoir_size-1])

        elif self.reservoir_connectivity == "sausage":
            J_ij_ = np.random.uniform(-self.gamma, self.gamma, (self.reservoir_size-1,))

            for i in range(self.reservoir_size-1):
                H_unitary_ += J_ij_[i]*(self.b_dag_system[i] @ self.b_system[i+1] + self.b_dag_system[i+1] @ self.b_system[i])

        self.H_unitary = H_unitary_
    
    def rk4_timesteps(self, rounds):
        """
        Declares the rounds of RK4 calculated for each input interaction.

        Parameters
        ----------
        rounds : Int value of the amount of rounds of RK4 for each interaction
        
        """

        self.step = 2*self.tau/rounds
        self.timesteps = np.arange(0, 2*self.tau, self.step)

    def inject_input(self, input):
        """
        Changes the input state interacting with the reservoir.
        Prints the coherence of the simulatin with Tr[rho].

        Parameters
        ----------
        input : Numpy array of the input state density matrix

        """
        rho_new_ = truncate_mantissa(tensor([input, partial_trace(self.rho_full, "second", self.dims[0], self.dims[1])]), self.sim_precision)

        print(f"tr_dm: {np.trace(rho_new_)}")

        self.rho_full = rho_new_ #truncate_mantissa(tensor([input, partial_trace(self.rho_full, "second", self.dims[0], self.dims[1])]), self.sim_precision)

    def update_reservoir(self):
        """
        Simulates the reservoir for one input state.
        Declares a piecewice function for reservoir interaction dynamics, and the RK4 fuction.

        """

        def update_me(rho, t):
            if 0 < t < self.tau:
                return self.A[1] @ rho + rho @ self.B[1] + sum([self.C[0][i] @ rho @ self.b_dag_system[i] + self.b_dag_system_P[i] @ rho @ self.b_system[i] for i in range(self.reservoir_size)]) + \
                    self.D[0] @ rho @ self.a_dag_system[0]
            elif self.tau < t < 2*self.tau:
                return self.A[2] @ rho + rho @ self.B[2] + sum([self.C[1][i] @ rho @ self.b_dag_system[i] + self.b_dag_system_P[i] @ rho @ self.b_system[i] for i in range(self.reservoir_size)]) + \
                    self.D[1] @ rho @ self.a_dag_system[1]
            elif (t == 0) | (t == self.tau) | (t == 2*self.tau):     
                return self.A[0] @ rho + rho @ self.B[0] + sum([self.b_system_gamma[i] @ rho @ self.b_dag_system[i] + self.b_dag_system_P[i] @ rho @ self.b_system[i] for i in range(self.reservoir_size)])
            else:
                warnings.warn("RK4 went out of bounds!", RuntimeWarning)
                return self.A[0] @ rho + rho @ self.B[0] + sum([self.b_system_gamma[i] @ rho @ self.b_dag_system[i] + self.b_dag_system_P[i] @ rho @ self.b_system[i] for i in range(self.reservoir_size)])

        def rk4(t):
            k1_ = update_me(self.rho_full, t)
            k2_ = update_me(self.rho_full + 0.5 * self.step * k1_, t + 0.5 * self.step)
            k3_ = update_me(self.rho_full + 0.5 * self.step * k2_, t + 0.5 * self.step)
            k4_ = update_me(self.rho_full + self.step * k3_, t + self.step)
            self.rho_full += copy.copy(truncate_mantissa((self.step/6)*(k1_ + 2 * k2_ + 2 * k3_ + k4_),self.sim_precision))

        for t in self.timesteps:
            rk4(t)
        
    def measure_reservoir(self):
        """
        Measures the reservoir average occupation numbers.

        """

        return [expectation_value(self.rho_full, self.b_dag_b[i]).real for i in range(self.reservoir_size)]
        
    def update_and_measure_reservoir(self, inputs):
        """
        Simulates the reservoir for a set on inputs and measures
        the average occupation numbers of the reservoir after each interaction.

        Parameters
        ----------
        inputs : A set of Numpy arrays of density matrices of a class of input states

        Returns
        -------
        A Numpy array of the measured observables

        """

        measured_observables_ = []
        for i,input in enumerate(inputs):
            self.inject_input(input)
            self.update_reservoir()
            measured_observables_.append(self.measure_reservoir())
            if i % 25 == 0:
                print(i)

        return np.array(measured_observables_)

    def assign_entanglement_from_probabilities(self, Y_pred_prob):
        """
        Assigns the entanglement values of inputs based on predictions.

        Parameters
        ----------
        Y_pred_prob : Numpy array of the predicted entanglement probabilities of input states

        Returns
        -------
        A Numpy array of the entanglement values
        
        """

        return np.array([[1,0] if x[0] >= x[1] else [0,1] for x in Y_pred_prob])

    def analyze_performance(self, Y_true, Y_pred):
        """
        Analyzes the performance of the entanglement classification task.

        Parameters
        ----------
        Y_true : Numpy array of the true entanglement values
        Y_pred : Numpy array of the predicted entanglement values

        Returns
        -------
        A tuple of the entanglement classification success rate and a list of confusion matrix values [TP, TN, FN, FP]
        
        """

        TP_ = 0
        TN_ = 0
        FN_ = 0
        FP_ = 0
        for x, y in zip(Y_true, Y_pred):
            if y[0] - x[0] == 0:
                if y[0] == 0:
                    TN_ += 1
                else:
                    TP_ += 1
            else:
                if y[0] == 0:
                    FN_ += 1
                else:
                    FP_ += 1

        return (TP_ + TN_) / len(Y_true), [TP_, TN_, FN_, FP_]

    def train_reservoir(self, inputs, entanglement_values):
        """
        Simulates the system for the training of the quantum reservoir.

        Parameters
        ----------
        inputs : A set of Numpy arrays of the input state density matrices
        entanglement_values : Numpy array of the true entanglement values

        """

        self.train_measured_observables = data_standardize(self.update_and_measure_reservoir(inputs))
        self.train_Y_true = entanglement_values
        self.rho_full_after_train = self.rho_full
        self.entangled_forecast.fit(self.train_measured_observables, self.train_Y_true[:,0])
        self.separable_forecast.fit(self.train_measured_observables, self.train_Y_true[:,1])

    def test_reservoir(self, inputs, entanglement_values):
        """
        Simulates the system for the testing of the quantum reservoir.

        Parameters
        ----------
        inputs : A set of Numpy arrays of the input state density matrices
        entanglement_values : Numpy array of the true entanglement values

        Returns
        -------
        A tuple of the entanglement classification success rate and a list of confusion matrix values [TP, TN, FN, FP]

        """

        self.test_measured_observables = data_standardize(self.update_and_measure_reservoir(inputs))
        self.test_Y_true = entanglement_values

        self.test_prob_Y_pred = np.array([self.entangled_forecast.predict(self.test_measured_observables), self.separable_forecast.predict(self.test_measured_observables)])
        self.test_Y_pred = self.assign_entanglement_from_probabilities(self.test_prob_Y_pred.T)
        #self.test_Y_pred_ = self.assign_entanglement_from_probabilities(np.array([self.entangled_forecast.predict(self.test_measured_observables_), self.separable_forecast.predict(self.test_measured_observables_)]).T)

        return self.analyze_performance(self.test_Y_true, self.test_Y_pred)
    
    def reset_after_test(self):
        """
        Resets the reservoir state to the one after training

        """

        self.rho_full = self.rho_full_after_train

    def system_save(self, filepath):
        """
        Saves the information of a trained model to a .npz file.

        Parameters
        ----------
        filepath : String value of the filepath

        """

        reservoir_params = {
        "gamma": self.gamma,
        "res_size": self.reservoir_size,
        "energy_trunc": self.energy_truncate_level,
        "dim_in": self.dims[0],
        "dim_res": self.dims[1],
        "res_connectivity": self.reservoir_connectivity,
        "sim_prec": self.sim_precision,
        "step": self.step,
        "steps": self.timesteps,
        "ridge_param": self.Ridge_param
        }

        np.savez(filepath, reservoir_params=reservoir_params, w_in=self.W_in, h_unit=self.H_unitary, rho_full=self.rho_full, train_obs=self.train_measured_observables, y_true=self.train_Y_true)

    def system_load(self, filepath):
        """
        Loads the information of a trained model from a .npz file.
        
        Parameters
        ----------
        filepath : String value of the filepath

        """
        data = np.load(filepath, allow_pickle=True)

        reservoir_params = data["reservoir_params"].item()
        self.W_in = data["w_in"]
        self.H_unitary = data["h_unit"]
        self.rho_full = data["rho_full"]
        self.train_measured_observables = data["train_obs"]
        self.train_Y_true = data["y_true"]

        self.gamma = reservoir_params["gamma"]
        self.reservoir_size = reservoir_params["res_size"]
        self.energy_truncate_level = reservoir_params["energy_trunc"]
        self.dims = (reservoir_params["dim_in"], reservoir_params["dim_res"])
        self.reservoir_connectivity = reservoir_params["res_connectivity"]
        self.sim_precision = reservoir_params["sim_prec"]
        self.step = reservoir_params["step"]
        self.timesteps = reservoir_params["steps"]
        self.Ridge_param = reservoir_params["ridge_param"]

        self.P = 0.1*self.gamma
        self.eta = sum(self.W_in**2)
        self.tau = 1/self.gamma
        self.a = init_destroy(self.energy_truncate_level)
        self.a_id = init_identity(self.energy_truncate_level)
        self.b = init_destroy(2)
        self.b_id = init_identity(2)
        system_dops_, system_cops_ = init_multipartite_dc_operators([self.a,self.b],[2,self.reservoir_size])
        self.a_system, self.a_dag_system, self.b_system, self.b_dag_system = system_dops_[0], system_cops_[0], system_dops_[1], system_cops_[1]
        self.A = np.array([-1j * self.H_unitary + sum([- 0.5 * self.gamma * b_dag @ b - 0.5 * self.P * b @ b_dag for b, b_dag in zip(self.b_system,self.b_dag_system)]),
                           *[-1j * self.H_unitary + sum([- 0.5 * self.gamma * b_dag @ b - 0.5 * self.P * b @ b_dag - w_in * b_dag @ a for b, b_dag, w_in in zip(self.b_system,self.b_dag_system,self.W_in)]) - \
                             0.5 * (self.eta/self.gamma) * a_dag @ a for a, a_dag in zip(self.a_system,self.a_dag_system)]])
        self.B = np.array([1j * self.H_unitary + sum([- 0.5 * self.gamma * b_dag @ b - 0.5 * self.P * b @ b_dag for b, b_dag in zip(self.b_system,self.b_dag_system)]),
                           *[1j * self.H_unitary + sum([- 0.5 * self.gamma * b_dag @ b - 0.5 * self.P * b @ b_dag - w_in * a_dag @ b for b, b_dag, w_in in zip(self.b_system,self.b_dag_system,self.W_in)]) - \
                             0.5 * (self.eta/self.gamma) * a_dag @ a for a, a_dag in zip(self.a_system,self.a_dag_system)]])
        self.C = np.array([[self.gamma * b + w_in * a for b, w_in in zip(self.b_system,self.W_in)] for a in self.a_system])
        self.D = np.array([sum([w_in * b for b, w_in in zip(self.b_system,self.W_in)]) + (self.eta/self.gamma) * a for a in self.a_system])
        self.b_system_gamma = np.array([self.gamma * b for b in self.b_system])
        self.b_dag_system_P = np.array([0.1 * self.gamma * b_dag for b_dag in self.b_dag_system])
        self.b_dag_b = np.array([b_dag @ b for b, b_dag in zip(self.b_system,self.b_dag_system)])
        self.entangled_forecast = Ridge(self.Ridge_param)
        self.separable_forecast = Ridge(self.Ridge_param)

        self.entangled_forecast.fit(self.train_measured_observables, self.train_Y_true[:,0])
        self.separable_forecast.fit(self.train_measured_observables, self.train_Y_true[:,1])

        self.rho_full_after_train = self.rho_full

    @classmethod
    def init_from_file(self, filepath):
        """
        Class method to initialize the QReservoir object from a .npz file.

        Parameters
        ----------
        filepath : String value of the filepath

        Returns
        -------
        A QReservoir object from the file

        """
        
        new_reservior_ = QReservoir()
        new_reservior_.system_load(filepath)

        return new_reservior_