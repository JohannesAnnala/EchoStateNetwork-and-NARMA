import numpy as np
from numpy.typing import NDArray
from tools import NMSE

class ESN:
    """
    A class for running an Echo State Network (ESN). The class uses a NARMA benchmarking task
    as performance metric. The NARMA task is evaluated with normalized mean squared error (NMSE).

    Attributes
    ----------
    reservoir_size : The amount of modes in the reservoir
    feedback_gain : Float value dictating the importance of past values
    input_gain : Float value dictating the importance of new values
    spatial_multiplexing : The amount of ensembles trained concurrently
    matrix_W : 2D Numpy array of the internal weights of the reservoir network
    vector_b : 1D Numpy array of a constant bias vector
    vector_v : 1D Numpy array of the input weights
    task_counter : The amount of different benchmarking tasks
    results : Keeps information of the bechmarking results
    prep_inputs : Inputs used in the preparation of the reservoir and benchmarking task
    train_inputs : Inputs used in the training of the reservoir
    test_inputs : Inputs used in the testing of the reservoir
    reservoir_state : The current state of the reservoir or reservoirs if spatial multiplexing is used
    trained_weights : The trained output weights of the ESN

    Methods
    -------
    generate_scaling() : Generates random values to fill matrix_W, vector_b and vector_v
    reset_reservoir_state() : Resets reservoir_state
    change_reservoir_constants() : Change constant reservoir values
    add_performance_task() : Adds a performance task for the reservoir
    delete_performance_task() : Deletes a reservoir performance task
    generate_inputs() : Generates a set of preparation, training and testing inputs
    measure_observables() : Measures the current state of the reservoir
    update_reservoir() : Updates the state of the reservoir for new inputs
    update_and_measure_reservoir() : Updates and measures the reservoir for a set of preparation, training or testing inputs
    NMSE() : Calculates the mean squared error of true and predicted outputs
    evolve_reservoir() : Performs the whole machine learning algorithm of preparation, training, testing and evaluation of an ESN with some performance task
    """

    def __init__(self, reservoir_size : int = 4, feedback_gain : np.float64 = 0.9, input_gain : np.float64 = 0.05, spatial_multiplexing : int = 1):
        self.reservoir_size = reservoir_size
        self.feedback_gain = feedback_gain
        self.input_gain = input_gain
        self.spatial_multiplexing = spatial_multiplexing
        
        self.generate_scaling()
        self.reset_reservoir_state()

        self.task_counter = 0
        self.results = {}

    def generate_scaling(self) -> None:
        """Generates random values to fill matrix_W, vector_b and vector_v."""

        scaled_W_ = [(1 / np.abs(np.linalg.eigvals(x)).max()) * x for x in [np.random.uniform(-1,1,(self.reservoir_size,self.reservoir_size)) for _ in range(self.spatial_multiplexing)]]

        self.matrix_W = np.zeros((self.reservoir_size*self.spatial_multiplexing,self.reservoir_size*self.spatial_multiplexing))
        for i in range(self.spatial_multiplexing):
            self.matrix_W[i*self.reservoir_size:(i+1)*self.reservoir_size,i*self.reservoir_size:(i+1)*self.reservoir_size] = scaled_W_[i]

        self.vector_b = np.random.uniform(-1,1,(self.reservoir_size*self.spatial_multiplexing,))
        self.vector_v = np.random.uniform(-1,1,(self.reservoir_size*self.spatial_multiplexing,))

    def reset_reservoir_state(self) -> None:
        """Resets the reservoir state to a random initial state"""

        self.reservoir_state = np.random.uniform(-1,1,(self.reservoir_size*self.spatial_multiplexing,))

    def change_reservoir_constants(self, reservoir_size : int = None, feedback_gain : np.float64 = None, input_gain : np.float64 = None, spatial_multiplexing : int = None) -> None:
        """Changes the constant values regarding the reservoir"""

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

    def add_performance_task(self, performance_task) -> None:
        """
        Adds a new performance task for the ESN to benchmark the performance

        Parameters
        ----------
        performance_task : The new performance task
        """

        self.task_counter += 1
        taskname_ = f"task{self.task_counter}"
        self.results[taskname_] = performance_task
        #print(f"Added performance task '{performance_task_.name}' as '{taskname_}'")

    def delete_performance_task(self, taskname : str) -> None:
        """
        Deletes a performance task from the ESN

        Parameters
        ----------
        taskname : A string value of the name of the performance task
        """

        del self.results[taskname]


    def generate_inputs(self, prep_train_test_split : list[int]) -> None:
        """
        Generates preparation, training and testing inputs for the ESN

        Parameters
        ----------
        prep_train_test_split : Size 3 list of the amounts of inputs associated with each phase
        """

        self.prep_inputs = np.random.uniform(-1,1,(prep_train_test_split[0],))
        self.train_inputs = np.random.uniform(-1,1,(prep_train_test_split[1],))
        self.test_inputs = np.random.uniform(-1,1,(prep_train_test_split[2],))

    def measure_observables(self) -> list[np.float64]:
        """
        Measures the current state of the reservoir

        Returns
        -------
        A list of the current state of the reservoir
        """

        return self.reservoir_state.tolist()

    def update_reservoir(self, input : np.float64 | np.int_) -> None:
        """
        Updates the state of the reservoir for one input

        Parameters
        ----------
        input : The value used in the updating of the reservoir
        """

        self.reservoir_state = np.tanh(self.feedback_gain * self.matrix_W @ self.reservoir_state + self.vector_b + self.input_gain * input * self.vector_v)

    def update_and_measure_reservoir(self, phase: str) -> NDArray[np.float64] | None:
        """
        Updates and measures the reservoir for a set of inputs.

        Parameters
        ----------
        phase : A string declaring if the updating happens in the preparation, training or testing phase

        Returns (if phase is train/true)
        -------
        A 2D numpy array of the measured reservoir states for all input values.
        """

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
    
    def evolve_reservoir(self, task: str, phase: str) -> None:
        """
        Performs a comprehensive simulation of the ESN for all possible phases.
        Saves the results as a tuple under the taskname.

        Parameters
        ----------
        task : Dictates which task is used as a performance metric
        phase : Chooses the phase of the simulation to be performed
        """

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

            self.results[task] = (self.results[task].name, NMSE(test_NARMA_outputs_, test_outputs_))

        elif phase == "prep":
            self.results[task].run(self.prep_inputs)
            self.update_and_measure_reservoir("prep") 
        elif phase == "train":
            train_observables_ = self.update_and_measure_reservoir("train")
            train_NARMA_outputs_ = self.results[task].run(self.train_inputs)
            self.trained_weights = np.linalg.inv(train_observables_.T @ train_observables_) @ train_observables_.T @ train_NARMA_outputs_
        elif phase == "test":
            test_observables_ = self.update_and_measure_reservoir("test")
            test_outputs_ = test_observables_ @ self.trained_weights
            test_NARMA_outputs_ = self.results[task].run(self.test_inputs)

            self.results[task] = (self.results[task].name, NMSE(test_NARMA_outputs_, test_outputs_))
        else:
            print("No such phase exists!")    
