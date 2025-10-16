import numpy as np

class Reservoir:

    #Generates various constant matrices and vectors used in the updation of reservoir
    def generate_scaling(self):
        #Matrix W is scaled to have spectral radius of 1
        unscaled_W_ = np.random.uniform(-1,1,(self.reservoir_size,self.reservoir_size))
        scaling_factor_W_ = 1 / np.abs(np.linalg.eigvals(unscaled_W_)).max()
        self.matrix_W = scaling_factor_W_*unscaled_W_

        self.vector_b = np.random.uniform(-1,1,(self.reservoir_size,))
        self.vector_v = np.random.uniform(-1,1,(self.reservoir_size,))

    #Resets the reservoir state to a random initial state
    def reset_reservoir_state(self):
        if self.spatial_multiplexing:
            self.reservoir_state = np.random.uniform(-1,1,(self.spatial_multiplexing,self.reservoir_size))
        else:
            self.reservoir_state = np.random.uniform(-1,1,(self.reservoir_size,))

    def __init__(self, reservoir_size_=4, feedback_gain_=0.9, input_gain_=0.1, spatial_multiplexing_=None):
        self.reservoir_size = reservoir_size_
        self.feedback_gain = feedback_gain_
        self.input_gain = input_gain_
        if spatial_multiplexing_:
            self.spatial_multiplexing = spatial_multiplexing_
        else:
            self.spatial_multiplexing = False
        
        self.generate_scaling()
        self.reset_reservoir_state()

        self.task_counter = 0
        self.results = {}

    #Updates the echo state network constants
    def update_reservoir_constant(self, reservoir_size_=None, feedback_gain_=None, input_gain_=None, spatial_multiplexing_=None):
        if reservoir_size_:
            self.reservoir_size = reservoir_size_
            self.generate_scaling()
        if feedback_gain_:
            self.feedback_gain = feedback_gain_
        if input_gain_:
            self.input_gain = input_gain_
        if spatial_multiplexing_:
            self.spatial_multiplexing = spatial_multiplexing_

        if reservoir_size_ | feedback_gain_ | input_gain_ | spatial_multiplexing_:
            self.reset_reservoir_state()

    def add_performance_task(self, performance_task_):
        self.task_counter += 1
        taskname_ = f"task{self.task_counter}"
        self.results[taskname_] = performance_task_
        print(f"Added performance task '{performance_task_.name}' as '{taskname_}'")

    def delete_performance_task(self, taskname_):
        del self.results[taskname_]

    #Generates inputs for reservoir preparation, training and testing
    def generate_inputs(self, prep_train_test_split_):
        self.prep_inputs = np.random.uniform(-1,1,(prep_train_test_split_[0],))
        self.train_inputs = np.random.uniform(-1,1,(prep_train_test_split_[1],))
        self.test_inputs = np.random.uniform(-1,1,(prep_train_test_split_[2],))

    def measure_observables(self):
        if not self.spatial_multiplexing:
            return self.reservoir_state.tolist()
        else:
            return np.concatenate(self.reservoir_state).tolist()

    #Updates reservoir state
    def update_reservoir(self, input_):
        if not self.spatial_multiplexing:
            self.reservoir_state = np.tanh(self.feedback_gain * self.matrix_W @ self.reservoir_state + self.vector_b + self.input_gain * input_ * self.vector_v)
        else:
            for i, network in enumerate(self.reservoir_state):
                self.reservoir_state[i] = np.tanh(self.feedback_gain * self.matrix_W @ network + self.vector_b + self.input_gain * input_ * self.vector_v)

    #Updates reservoir state and measures it's observables
    def update_and_measure_reservoir(self, phase_: str):
        if phase_ == "prep":
            for prep_input in self.prep_inputs:
                self.update_reservoir(prep_input)

        elif phase_ == "train":
            measured_observables_ = []
            for train_input in self.train_inputs:
                self.update_reservoir(train_input)
                measured_observables_.append(self.measure_observables() + [1])
            return np.array(measured_observables_)
        
        elif phase_ == "test":
            measured_observables_ = []
            for test_input in self.test_inputs:
                self.update_reservoir(test_input)
                measured_observables_.append(self.measure_observables() + [1])
            return np.array(measured_observables_)

    def NMSE(self, true_outputs_, generated_outputs_):
        return sum([(x-y)**2 for x,y in zip(true_outputs_, generated_outputs_)])/sum([x**2 for x in true_outputs_])
    
    def evolve_reservoir(self, task_: str, phase_: str):

        #Full simulation
        if phase_ == "full":
            #Preparation
            self.results[task_].perform_NARMA(self.prep_inputs)
            self.update_and_measure_reservoir("prep")

            #Training
            train_observables_ = self.update_and_measure_reservoir("train")
            train_NARMA_outputs_ = self.results[task_].perform_NARMA(self.train_inputs)
            self.trained_weights = np.linalg.inv(train_observables_.T @ train_observables_) @ train_observables_.T @ train_NARMA_outputs_

            #Testing
            test_observables_ = self.update_and_measure_reservoir("test")
            test_outputs_ = test_observables_ @ self.trained_weights
            test_NARMA_outputs_ = self.results[task_].perform_NARMA(self.test_inputs)

            self.results[task_] = (self.results[task_].name, self.NMSE(test_NARMA_outputs_, test_outputs_))

        #Preparation phase, where the dependence on the initial state of reservoir is removed 
        elif phase_ == "prep":
            self.results[task_].perform_NARMA(self.prep_inputs)
            self.update_and_measure_reservoir("prep") 
        #Training phase, where the weights are trained
        elif phase_ == "train":
            train_observables_ = self.update_and_measure_reservoir("train")
            train_NARMA_outputs_ = self.results[task_].perform_NARMA(self.train_inputs)
            self.trained_weights = np.linalg.inv(train_observables_.T @ train_observables_) @ train_observables_.T @ train_NARMA_outputs_
        #Testing phase
        elif phase_ == "test":
            test_observables_ = self.update_and_measure_reservoir("test")
            test_outputs_ = test_observables_ @ self.trained_weights
            test_NARMA_outputs_ = self.results[task_].perform_NARMA(self.test_inputs)

            self.results[task_] = (self.results[task_].name, self.NMSE(test_NARMA_outputs_, test_outputs_))
        else:
            print("No such phase exists!")    
