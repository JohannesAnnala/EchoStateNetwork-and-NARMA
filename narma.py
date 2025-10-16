
class NARMA:
    def __init__(self, degree_=2, alpha_=0.3, beta_=0.05, gamma_=0.375, delta_=0):
        self.alpha = alpha_
        self.beta = beta_
        self.gamma = gamma_
        self.delta = delta_
        self.NARMA_degree = degree_
        self.NARMA_outputs = [0] * degree_
        self.NARMA_inputs = [0] * degree_

        self.name = f'NARMA({degree_}, {alpha_}, {beta_}, {gamma_}, {delta_})'

    def update_NARMA_constant(self, alpha_=None, beta_=None, gamma_=None, delta_=None):
        if alpha_:
            self.alpha = alpha_
        if beta_:
            self.beta = alpha_
        if gamma_:
            self.gamma = gamma_
        if delta_:
            self.delta = delta_

    def update_NARMA_degree(self, degree_):
        if degree_ % 1 == 0:
            self.NARMA_degree = degree_
            self.NARMA_outputs = [0] * degree_
            self.NARMA_inputs = [0] * degree_
        else:
            print("Degree has to be a whole number")

    def reset_NARMA(self):
        self.NARMA_outputs = [0] * self.NARMA_degree
        self.NARMA_inputs = [0] * self.NARMA_degree

    def perform_NARMA(self, inputs_):
        NARMA_vals_ = []
        for input in inputs_:
            NARMA_new_value_ = self.alpha * self.NARMA_outputs[0] + self.beta * self.NARMA_outputs[0] * sum(self.NARMA_outputs) + self.gamma * self.NARMA_inputs[-1] * self.NARMA_inputs[0] + self.delta

            NARMA_vals_.append(NARMA_new_value_)
            self.NARMA_outputs = [NARMA_new_value_] + self.NARMA_outputs[:-1]
            self.NARMA_inputs = [input] + self.NARMA_inputs[:-1]
    
        return NARMA_vals_
    