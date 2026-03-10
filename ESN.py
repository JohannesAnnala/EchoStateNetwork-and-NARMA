import numpy as np

class ESN:

    #Generates various constant matrices and vectors used in the updation of reservoir
    def generate_scaling(self):

        #Matrix W is scaled to have spectral radius of 1
        scaled_W_ = [(1 / np.abs(np.linalg.eigvals(x)).max()) * x for x in [np.random.uniform(-1,1,(self.reservoir_size,self.reservoir_size)) for _ in range(self.spatial_multiplexing)]]

        self.matrix_W = np.zeros((self.reservoir_size*self.spatial_multiplexing,self.reservoir_size*self.spatial_multiplexing))
        for i in range(self.spatial_multiplexing):
            self.matrix_W[i*self.reservoir_size:(i+1)*self.reservoir_size,i*self.reservoir_size:(i+1)*self.reservoir_size] = scaled_W_[i]

        self.vector_b = np.random.uniform(-1,1,(self.reservoir_size*self.spatial_multiplexing,))
        self.vector_v = np.random.uniform(-1,1,(self.reservoir_size*self.spatial_multiplexing,))

    #Resets the reservoir state to a random initial state
    def reset_reservoir_state(self):
        self.reservoir_state = np.random.uniform(-1,1,(self.reservoir_size*self.spatial_multiplexing,))

    def __init__(self, reservoir_size=4, feedback_gain=0.9, input_gain=0.05, spatial_multiplexing=1):
        self.reservoir_size = reservoir_size
        self.feedback_gain = feedback_gain
        self.input_gain = input_gain
        self.spatial_multiplexing = spatial_multiplexing
        
        self.generate_scaling()
        self.reset_reservoir_state()

        self.task_counter = 0
        self.results = {}

    #Updates the echo state network constants
    def update_reservoir_constant(self, reservoir_size=None, feedback_gain=None, input_gain=None, spatial_multiplexing=None):
        if reservoir_size:
            self.reservoir_size = reservoir_size
            self.generate_scaling()
        if feedback_gain:
            self.feedback_gain = feedback_gain
        if input_gain:
            self.input_gain = input_gain
        if spatial_multiplexing:
            self.spatial_multiplexing = spatial_multiplexing
            self.generate_scaling()

        if reservoir_size | feedback_gain | input_gain | spatial_multiplexing:
            self.reset_reservoir_state()

    def add_performance_task(self, performance_task):
        self.task_counter += 1
        taskname_ = f"task{self.task_counter}"
        self.results[taskname_] = performance_task
        #print(f"Added performance task '{performance_task_.name}' as '{taskname_}'")

    def delete_performance_task(self, taskname):
        del self.results[taskname]

    #Generates inputs for reservoir preparation, training and testing
    def generate_inputs(self, prep_train_test_split):
        self.prep_inputs = np.random.uniform(-1,1,(prep_train_test_split[0],))
        self.train_inputs = np.random.uniform(-1,1,(prep_train_test_split[1],))
        self.test_inputs = np.random.uniform(-1,1,(prep_train_test_split[2],))

    def measure_observables(self):
        return self.reservoir_state.tolist()

    #Updates reservoir state
    def update_reservoir(self, input):
        self.reservoir_state = np.tanh(self.feedback_gain * self.matrix_W @ self.reservoir_state + self.vector_b + self.input_gain * input * self.vector_v)

    #Updates reservoir state and measures it's observables
    def update_and_measure_reservoir(self, phase: str):
        if phase == "prep":
            for prep_input in self.prep_inputs:
                self.update_reservoir(prep_input)

        elif phase == "train":
            measured_observables_ = []
            for train_input in self.train_inputs:
                self.update_reservoir(train_input)
                measured_observables_.append(self.measure_observables() + [1])
            return np.array(measured_observables_)
        
        elif phase == "test":
            measured_observables_ = []
            for test_input in self.test_inputs:
                self.update_reservoir(test_input)
                measured_observables_.append(self.measure_observables() + [1])
            return np.array(measured_observables_)

    def NMSE(self, true_outputs, generated_outputs):
        return sum([(x-y)**2 for x,y in zip(true_outputs, generated_outputs)])/sum([x**2 for x in true_outputs])
    
    def evolve_reservoir(self, task: str, phase: str):

        #Full simulation
        if phase == "full":
            #Preparation
            self.results[task].run(self.prep_inputs)
            self.update_and_measure_reservoir("prep")

            #Training
            train_observables_ = self.update_and_measure_reservoir("train")
            train_NARMA_outputs_ = self.results[task].run(self.train_inputs)

            self.trained_weights = np.linalg.inv(train_observables_.T @ train_observables_) @ train_observables_.T @ train_NARMA_outputs_

            #Testing
            test_observables_ = self.update_and_measure_reservoir("test")
            test_outputs_ = test_observables_ @ self.trained_weights
            test_NARMA_outputs_ = self.results[task].run(self.test_inputs)

            self.results[task] = (self.results[task].name, self.NMSE(test_NARMA_outputs_, test_outputs_))

        #Preparation phase, where the dependence on the initial state of reservoir is removed 
        elif phase == "prep":
            self.results[task].run(self.prep_inputs)
            self.update_and_measure_reservoir("prep") 
        #Training phase, where the weights are trained
        elif phase == "train":
            train_observables_ = self.update_and_measure_reservoir("train")
            train_NARMA_outputs_ = self.results[task].run(self.train_inputs)
            self.trained_weights = np.linalg.inv(train_observables_.T @ train_observables_) @ train_observables_.T @ train_NARMA_outputs_
        #Testing phase
        elif phase == "test":
            test_observables_ = self.update_and_measure_reservoir("test")
            test_outputs_ = test_observables_ @ self.trained_weights
            test_NARMA_outputs_ = self.results[task].run(self.test_inputs)

            self.results[task] = (self.results[task].name, self.NMSE(test_NARMA_outputs_, test_outputs_))
        else:
            print("No such phase exists!")    
