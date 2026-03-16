import numpy as np
from itertools import combinations
from tools import init_identity, tensor, dagger, init_destroy, init_multipartite_dc_operators, init_vac, assess_dm_entanglement, truncate_mantissa, data_standardize
from ridge import RidgeRegression

from sklearn.linear_model import Ridge
import copy
import warnings

warnings.simplefilter("always", RuntimeWarning)

class QReservoir:
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

    def update_gamma(self, gamma):
        self.gamma = gamma
        self.P = 0.1*gamma
        self.W_in = np.random.uniform(0,gamma,(self.reservoir_size,))
        self.eta = sum(self.W_in**2)
        self.tau = 1/gamma

    def update_energy_truncate_level(self, truncate):
        self.energy_truncate_level = truncate

        self.a = init_destroy(truncate)
        self.a_id = init_identity(truncate)

        new_dops_, new_cops_ = init_multipartite_dc_operators([self.a, self.b], [2, self.reservoir_size])
        self.a_system, self.b_system, self.b_dag_system = new_dops_[0], new_dops_[1], new_cops_[1] 

    def update_reservoir_size(self, reservoir_size): 
        self.reservoir_size = reservoir_size      

        new_dops_, new_cops_ = init_multipartite_dc_operators([self.a, self.b], [2, self.reservoir_size])
        self.a_system, self.b_system, self.b_dag_system = new_dops_[0], new_dops_[1], new_cops_[1] 
    
    def update_reservoir_connectivity(self, reservoir_connectivity):
        self.reservoir_connectivity = reservoir_connectivity
        self.init_unitary_evolution()

    def init_reservoir(self, initial_state):
        if initial_state == "vacuum":

            self.rho_full = init_vac(self.dims[0] * self.dims[1])
        else:
            raise ValueError("Quantum reservoir can't be initialized in such an initial state, try 'vacuum'")

    def init_unitary_evolution(self):

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
    
    def rk4_timesteps(self, timesteps):
        self.step = 2*self.tau/timesteps
        self.timesteps = np.arange(0, 2*self.tau, self.step)

    #Changes the input state interacting with the reservoir
    def inject_input(self, input):
        rho_new_ = np.zeros((self.dims[1],self.dims[1]), dtype=np.complex64)
        for i in range(self.dims[0]):
            #buffer = self.rho_full[i*self.dims[1]:(i+1)*self.dims[1], i*self.dims[1]:(i+1)*self.dims[1]]
            #print(np.round(buffer,3))
            rho_new_ += self.rho_full[i*self.dims[1]:(i+1)*self.dims[1], i*self.dims[1]:(i+1)*self.dims[1]]
   
        buffer = truncate_mantissa(np.kron(input, rho_new_),self.sim_precision)
        tr_res = np.trace(rho_new_)
        tr_new = np.trace(buffer)
        print(f"tr_res: {tr_res}")
        print(f"tr_new: {tr_new}")

        self.rho_full = buffer #truncate_mantissa(np.kron(input, rho_new_),self.sim_precision)

    def update_reservoir(self):

        def update_me(rho, t):
            
            if 0 < t < self.tau:
                #print("eka")
                return self.A[1] @ rho + rho @ self.B[1] + sum([self.C[0][i] @ rho @ self.b_dag_system[i] + self.b_dag_system_P[i] @ rho @ self.b_system[i] for i in range(self.reservoir_size)]) + \
                    self.D[0] @ rho @ self.a_dag_system[0]
            elif self.tau < t < 2*self.tau:
                #print("toka")
                return self.A[2] @ rho + rho @ self.B[2] + sum([self.C[1][i] @ rho @ self.b_dag_system[i] + self.b_dag_system_P[i] @ rho @ self.b_system[i] for i in range(self.reservoir_size)]) + \
                    self.D[1] @ rho @ self.a_dag_system[1]
            elif (t == 0) | (t == self.tau) | (t == 2*self.tau):
                #print(f"t: {t}, valiin")       
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
        return [np.trace(self.rho_full @ self.b_dag_b[i]).real for i in range(self.reservoir_size)]
        
    def update_and_measure_reservoir(self, inputs):
        measured_observables_ = []
        for i,input in enumerate(inputs):
            self.inject_input(input)
            self.update_reservoir()
            measured_observables_.append(self.measure_reservoir())
            if i % 25 == 0:
                print(i)

        return np.array(measured_observables_)

    def assign_entanglement_from_probabilities(self, Y_pred):
        return np.array([[1,0] if x[0] >= x[1] else [0,1] for x in Y_pred])

    def analyze_performance(self, Y_true, Y_pred):
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
        self.train_measured_observables = data_standardize(self.update_and_measure_reservoir(inputs))
        self.train_Y_true = entanglement_values
        self.rho_full_after_train = self.rho_full
        self.entangled_forecast.fit(self.train_measured_observables, self.train_Y_true[:,0])
        self.separable_forecast.fit(self.train_measured_observables, self.train_Y_true[:,1])

    def test_reservoir(self, inputs, entanglement_values):
        self.test_measured_observables = data_standardize(self.update_and_measure_reservoir(inputs))
        self.test_Y_true = entanglement_values

        self.test_prob_Y_pred = np.array([self.entangled_forecast.predict(self.test_measured_observables), self.separable_forecast.predict(self.test_measured_observables)])
        self.test_Y_pred = self.assign_entanglement_from_probabilities(self.test_prob_Y_pred.T)
        #self.test_Y_pred_ = self.assign_entanglement_from_probabilities(np.array([self.entangled_forecast.predict(self.test_measured_observables_), self.separable_forecast.predict(self.test_measured_observables_)]).T)

        return self.analyze_performance(self.test_Y_true, self.test_Y_pred)
    
    def reset_after_test(self):
        self.rho_full = self.rho_full_after_train

    def system_save(self, filepath):
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
        
        new_reservior_ = QReservoir()
        new_reservior_.system_load(filepath)

        return new_reservior_
