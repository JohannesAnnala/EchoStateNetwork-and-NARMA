import numpy as np
from itertools import combinations
from tools import init_identity, tensor, dagger, init_destroy, init_multipartite_dc_operators, expectation_value, assess_dm_entanglement
from ridge import RidgeRegression

from sklearn.linear_model import Ridge
import copy

class QReservoir:
    def __init__(self, gamma, reservoir_size=4, energy_truncate_level=5, reservoir_connectivity="alltoall"):
        self.reservoir_size = reservoir_size
        self.energy_truncate_level = energy_truncate_level
        self.dims = (energy_truncate_level**2, 2**reservoir_size)
        self.size = (energy_truncate_level**2)*(2**reservoir_size)
        self.reservoir_connectivity = reservoir_connectivity
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

        self.entangled_forecast = Ridge(1.0)
        self.separable_forecast = Ridge(1.0)

        #self.entangled_forecast = RidgeRegression(0.001, 0.1, 1000)
        #self.separable_forecast = RidgeRegression(0.001, 0.1, 1000)

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
            rho_init_ = np.zeros((2**self.reservoir_size, 2**self.reservoir_size), dtype=np.complex64)
            rho_init_[0,0] = 1

            input_init_ = np.zeros((self.energy_truncate_level**2, self.energy_truncate_level**2))
            input_init_[0,0] = 1

            self.rho_full = tensor([input_init_, rho_init_])
        else:
            print(f"No such initial state as {initial_state}")

    def init_unitary_evolution(self):

        dims_ = self.dims[0] * self.dims[1]
        H_unitary_ = np.zeros((dims_, dims_))

        if self.reservoir_connectivity == "alltoall":
            J_ij_ = np.random.uniform(-self.gamma, self.gamma, int((self.reservoir_size*(self.reservoir_size-1)/2),))

            for (first, second), interaction_str in zip(combinations(range(self.reservoir_size), 2), J_ij_):
                H_unitary_ += interaction_str*(dagger(self.b_system[first]) @ self.b_system[second] + dagger(self.b_system[second]) @ self.b_system[first])

        elif self.reservoir_connectivity == "ring":
            J_ij = np.random.uniform(-self.gamma, self.gamma, (self.reservoir_size,))

        elif self.reservoir_connectivity == "sausage":
            J_ij = np.random.uniform(-self.gamma, self.gamma, (self.reservoir_size-1,))

        else:
            print("No can do")

        self.H_unitary = H_unitary_
    
    def rk4_timesteps(self, timesteps):
        self.step = 2*self.tau/timesteps
        self.timesteps = np.arange(0, 2*self.tau, self.step)



    #ALL OF THE MODIFIED FUNCTIONS ARE BELOW



    #Changes the input state interacting with the reservoir
    def inject_input(self, input):
        rho_new_ = np.zeros((self.dims[1],self.dims[1]), dtype=np.complex64)
        for i in range(self.dims[0]):
            #buffer = self.rho_full[i*self.dims[1]:(i+1)*self.dims[1], i*self.dims[1]:(i+1)*self.dims[1]]
            #print(np.round(buffer,3))
            rho_new_ += self.rho_full[i*self.dims[1]:(i+1)*self.dims[1], i*self.dims[1]:(i+1)*self.dims[1]]
   
        buffer = np.kron(input, rho_new_).round(8)

        tr_input = np.trace(input)
        tr_res = np.trace(rho_new_)
        tr_new = np.trace(buffer)
        print(f"tr_input: {tr_input}")
        print(f"tr_res: {tr_res}")
        print(f"tr_new: {tr_new}")

        tr_input = np.trace(input.round(8))
        tr_res = np.trace(rho_new_.round(8))
        tr_new = np.trace(buffer.round(8))
        print(f"tr_input: {tr_input}")
        print(f"tr_res: {tr_res}")
        print(f"tr_new: {tr_new}")

        self.rho_full = buffer

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
                return np.zeros((12,12))
                print("Out of bounds")

        def rk4(t):
            k1_ = update_me(self.rho_full, t)
            k2_ = update_me(self.rho_full + 0.5 * self.step * k1_, t + 0.5 * self.step)
            k3_ = update_me(self.rho_full + 0.5 * self.step * k2_, t + 0.5 * self.step)
            k4_ = update_me(self.rho_full + self.step * k3_, t + self.step)
            self.rho_full += (self.step/6)*(k1_ + 2 * k2_ + 2 * k3_ + k4_)

        for t in self.timesteps:
            rk4(t)
        
    def measure_reservoir(self):
        return [np.trace(self.rho_full @ b_dag_b) for b_dag_b in self.b_dag_b]
        
    def update_and_measure_reservoir(self, inputs):
        measured_observables_ = []
        self.stored_rho = [self.rho_full]
        for i,input in enumerate(inputs):
            self.inject_input(input)
            self.stored_rho.append(self.rho_full)
            self.update_reservoir()
            self.stored_rho.append(self.rho_full)
            measured_observables_.append(self.measure_reservoir())
            print(i)

        return np.array(measured_observables_)
    
    def get_entanglement_values(self, inputs):
        return np.array([[1,0] if assess_dm_entanglement(input, "first", self.energy_truncate_level, self.energy_truncate_level, 2) > 0 else [0,1] for input in inputs])

    def assign_entanglement_from_probabilities(self, Y_pred):
        return np.array([[1,0] if x[0] >= x[1] else [0,1] for x in Y_pred])

    def analyze_performance(self, Y_true, Y_pred):
        count_ = 0
        for x, y in zip(Y_true, Y_pred):
            if x[0] == y[0]:
                count_ += 1

        return count_ / len(Y_true)

    def train_reservoir(self, inputs):
        self.train_measured_observables_ = self.update_and_measure_reservoir(inputs)
        self.train_Y_true_ = self.get_entanglement_values(inputs)
        #self.entangled_forecast.fit(self.train_measured_observables_, self.train_Y_true_[:,0])
        #self.separable_forecast.fit(self.train_measured_observables_, self.train_Y_true_[:,1])

    def test_reservoir(self, inputs):
        self.test_measured_observables_ = self.update_and_measure_reservoir(inputs)
        self.test_Y_true_ = self.get_entanglement_values(inputs)

        self.test_pre_Y_pred_ = np.array([self.entangled_forecast.predict(self.test_measured_observables_), self.separable_forecast.predict(self.test_measured_observables_)])
        self.test_Y_pred_ = self.assign_entanglement_from_probabilities(self.test_pre_Y_pred_.T)

        score_ = self.analyze_performance(self.test_Y_true_, self.test_Y_pred_)

        print(score_)
        return score_

