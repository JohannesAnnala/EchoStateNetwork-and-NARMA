class NARMA:
    """
    A class for computing different degrees of nonlinear autoregressive moving average NARMA.

    Attributes
    ----------
    alpha : Real value used in the computation of NARMA
    beta : Real value used in the computation of NARMA
    gamma : Real value used in the computation of NARMA
    delta : Real value used in the computation of NARMA
    NARMA_degree : Natural number describing the degree of NARMA
    NARMA_inputs : List of past recent NARMA inputs
    NARMA_outputs : List of past recent NARMA outputs
    name : A string describing the hyperparameters of NARMA

    Methods
    -------
    update_NARMA_constant() : Updates the real constant values used in NARMA
    update_degree() : Updates the NARMA degree
    reset_NARMA() : Resets past NARMA inputs and outputs
    run() : Runs NARMA for a set of inputs

    """

    def __init__(self, degree=2, alpha=0.3, beta=0.05, gamma=0.375, delta=0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.NARMA_degree = degree
        self.NARMA_inputs = [0] * degree
        self.NARMA_outputs = [0] * degree

        self.name = f'NARMA({degree}, {alpha}, {beta}, {gamma}, {delta})'

    def update_NARMA_constant(self, alpha=None, beta=None, gamma=None, delta=None):
        """
        Updates constants used in the running of NARMA.

        Parameters
        ----------
        alpha : Real value used in the computation of NARMA
        beta : Real value used in the computation of NARMA
        gamma : Real value used in the computation of NARMA
        delta : Real value used in the computation of NARMA 

        """

        if alpha:
            self.alpha = alpha
        if beta:
            self.beta = alpha
        if gamma:
            self.gamma = gamma
        if delta:
            self.delta = delta
        
    def update_degree(self, degree):
        """
        Updates the degree of NARMA. Checks if the given value is a natural number.

        Parameters
        ----------
        degree : The new degree of NARMA

        """

        if degree % 1 == 0:
            self.NARMA_degree = degree
            self.reset_NARMA()
        else:
            print("Degree has to be a whole number")

    def reset_NARMA(self):
        """
        Resets the state of NARMA.

        """

        self.NARMA_inputs = [0] * self.NARMA_degree
        self.NARMA_outputs = [0] * self.NARMA_degree

    def run(self, inputs):
        """
        Runs NARMA for a set of given input values.

        Parameters
        ----------
        inputs : A list of given inputs used in the running of NARMA

        Returns
        -------
        NARMA_vals_ : All of the NARMA outputs produced during the running

        """
        
        NARMA_vals_ = []
        for input in inputs:
            NARMA_new_value_ = self.alpha * self.NARMA_outputs[0] + self.beta * self.NARMA_outputs[0] * sum(self.NARMA_outputs) + self.gamma * self.NARMA_inputs[-1] * self.NARMA_inputs[0] + self.delta

            NARMA_vals_.append(NARMA_new_value_)
            self.NARMA_outputs = [NARMA_new_value_] + self.NARMA_outputs[:-1]
            self.NARMA_inputs = [input] + self.NARMA_inputs[:-1]
    
        return NARMA_vals_