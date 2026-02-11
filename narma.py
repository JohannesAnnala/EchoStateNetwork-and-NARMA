
class NARMA:
    def __init__(self, degree=2, alpha=0.3, beta=0.05, gamma=0.375, delta=0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.NARMA_degree = degree
        self.NARMA_outputs = [0] * degree
        self.NARMA_inputs = [0] * degree

        self.name = f'NARMA({degree}, {alpha}, {beta}, {gamma}, {delta})'

    def update_NARMA_constant(self, alpha=None, beta=None, gamma=None, delta=None):
        if alpha:
            self.alpha = alpha
        if beta:
            self.beta = alpha
        if gamma:
            self.gamma = gamma
        if delta:
            self.delta = delta

    def update_degree(self, degree):
        if degree % 1 == 0:
            self.NARMA_degree = degree
            self.NARMA_outputs = [0] * degree
            self.NARMA_inputs = [0] * degree
        else:
            print("Degree has to be a whole number")

    def reset_NARMA(self):
        self.NARMA_outputs = [0] * self.NARMA_degree
        self.NARMA_inputs = [0] * self.NARMA_degree

    def run(self, inputs):
        NARMA_vals_ = []
        for input in inputs:
            NARMA_new_value_ = self.alpha * self.NARMA_outputs[0] + self.beta * self.NARMA_outputs[0] * sum(self.NARMA_outputs) + self.gamma * self.NARMA_inputs[-1] * self.NARMA_inputs[0] + self.delta

            NARMA_vals_.append(NARMA_new_value_)
            self.NARMA_outputs = [NARMA_new_value_] + self.NARMA_outputs[:-1]
            self.NARMA_inputs = [input] + self.NARMA_inputs[:-1]
    
        return NARMA_vals_
    