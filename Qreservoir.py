import numpy as np
from itertools import combinations
from tools import init_identity, tensor, dagger, partial_trace, rk4
from Qtools import init_destroy, init_multipartite_dc_operators, init_dissipator, init_dissipators, expectation_value, assess_dm_entanglement
from ridge import RidgeRegression

import time

class QReservoir:
    def __init__(self, gamma, reservoir_size=4, energy_truncate_level=5, reservoir_connectivity="alltoall"):
        self.reservoir_size = reservoir_size
        self.energy_truncate_level = energy_truncate_level
        self.dims = (energy_truncate_level**2, 2**reservoir_size)
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
        self.a_system, self.b_system, self.b_dag_system = system_dops_[0], system_dops_[1], system_cops_[1]

        self.entangled_forecast = RidgeRegression(0.001, 0.1, 1000)
        self.separable_forecast = RidgeRegression(0.001, 0.1, 1000)

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
            rho_init_ = np.zeros((2**self.reservoir_size, 2**self.reservoir_size), dtype=complex)
            rho_init_[0,0] = 1

            input_init_ = np.zeros((self.energy_truncate_level**2, self.energy_truncate_level**2))
            input_init_[0,0] = 1

            self.rho_full = tensor([input_init_, rho_init_])
        else:
            print(f"No such initial state as {initial_state}")

    #Changes the input state interacting with the reservoir
    def inject_input(self, input):
        self.rho_full = tensor([input, partial_trace(self.rho_full, "second", self.dims[0], self.dims[1])])

    def init_unitary_evolution(self):

        dims_ = self.dims[0] * self.dims[1]
        H_unitary_ = np.empty([dims_, dims_])

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
        
    def init_cascaded_interaction(self, rho, interacting_mode):
        return sum([input_str*(interacting_mode @ rho @ dagger(b) - dagger(b) @ interacting_mode @ rho + b @ rho @ dagger(interacting_mode) - rho @ dagger(interacting_mode) @ b) for input_str, b in zip(self.W_in, self.b_system)])

    def rk4_timesteps(self, timesteps):
        self.step = 2*self.tau/timesteps
        self.timesteps = np.arange(0, 2*self.tau, self.step)

    def update_reservoir(self):

        def update_me(rho, t):

            time_ = time.perf_counter()
            unitary_evolution_ = -1j * (self.H_unitary @ rho - rho @ self.H_unitary)
            print(f"unitary {time.perf_counter()-time_}")
            time_ = time.perf_counter()
            nonunitary_evolution_ = (self.gamma/2)*init_dissipators(rho, self.b_system) + (self.P/2)*init_dissipators(rho, self.b_dag_system)
            print(f"nonunitary {time.perf_counter()-time_}")

            if 0 < t < self.tau:
                time_ = time.perf_counter()
                nonunitary_evolution_ += self.init_cascaded_interaction(rho, self.a_system[0]) + (self.eta/(2*self.gamma))*init_dissipator(rho, self.a_system[0])
                print(f"1st mode {time.perf_counter()-time_}")
            elif self.tau < t < 2*self.tau:
                time_ = time.perf_counter()
                nonunitary_evolution_ += self.init_cascaded_interaction(rho, self.a_system[1]) + (self.eta/(2*self.gamma))*init_dissipator(rho, self.a_system[1])
                print(f"2nd mode {time.perf_counter()-time_}")
            return unitary_evolution_ + nonunitary_evolution_

        full_time_ = time.perf_counter()
        for t in self.timesteps:
            time_rk_ = time.perf_counter()
            self.rho_full = rk4(update_me, self.rho_full, t, self.step)
            print(f"rk4 step {t} took {time.perf_counter()-time_rk_}")
            print()
        print(f"update took {time.perf_counter()-full_time_}")
        
    def measure_reservoir(self):
        return [expectation_value(self.rho_full, self.b_dag_system[i] @ self.b_system[i]) for i in range(self.reservoir_size)]
        
    def update_and_measure_reservoir(self, inputs):
        measured_observables_ = []
        for i,input in enumerate(inputs):
            print(i)
            self.inject_input(input)
            self.update_reservoir()
            measured_observables_.append(self.measure_reservoir())
            print()

        return np.array(measured_observables_)
    
    def get_entanglement_values(self, inputs):
        return np.array([[1,0] if assess_dm_entanglement(input, "first", self.energy_truncate_level, self.energy_truncate_level) > 0 else [0,1] for input in inputs])

    def assign_entanglement_from_probabilities(self, Y_pred):
        return np.array([[1,0] if x[0] >= x[1] else [0,1] for x in Y_pred])

    def analyze_performance(self, Y_true, Y_pred):
        count_ = 0
        for x, y in zip(Y_true, Y_pred):
            if x[0] == y[0]:
                count_ += 1

        return count_ / len(Y_true)

    def train_reservoir(self, inputs):
        measured_observables_ = self.update_and_measure_reservoir(inputs)
        Y_true_ = self.get_entanglement_values(inputs)
        self.entangled_forecast.fit(measured_observables_, Y_true_[:,0])
        self.separable_forecast.fit(measured_observables_, Y_true_[:,1])

    def test_reservoir(self, inputs):
        measured_observables_ = self.update_and_measure_reservoir(inputs)
        Y_true_ = self.get_entanglement_values(inputs)

        Y_pred_ = np.array([self.entangled_forecast.predict(measured_observables_), self.separable_forecast.predict(measured_observables_)])
        Y_pred_ = self.assign_entanglement_from_probabilities(Y_pred_.T)

        score_ = self.analyze_performance(Y_true_, Y_pred_)

        return score_
