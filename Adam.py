import numpy as np
from numba import jit
from numba import cuda
cuda.select_device(0)

@jit
def optimizeConvJIT(m_k, v_k, beta_1, beta_2, kGradients, learnRate, epsilon, t, m_hat_k = 0, v_hat_k = 0):
    m_k = beta_1 * m_k + (1-beta_1) * kGradients
    v_k = beta_2 * v_k + (1-beta_2) * (kGradients**2)
    m_hat_k = (m_k) / (1-beta_1**t)
    v_hat_k = (v_k) / (1-beta_2**t)
    kernelsChange = learnRate * (m_hat_k / (np.sqrt(v_hat_k) + epsilon))

    return kernelsChange

@jit
def optimizeFCJIT(m_w, v_w, m_b, v_b, beta_1, beta_2, wGradients, bGradients, learnRate, epsilon, t, m_hat_w = 0, v_hat_w = 0, m_hat_b = 0, v_hat_b = 0):
    m_w = beta_1 * m_w + (1-beta_1) * wGradients
    v_w = beta_2 * v_w + (1-beta_2) * wGradients**2

    m_b = beta_1 * m_b + (1-beta_1) * bGradients
    v_b = beta_2 * v_b + (1-beta_2) * bGradients**2

    m_hat_w = (m_w) / (1-beta_1**t)
    v_hat_w = (v_w) / (1-beta_2**t)
    weightsChange = learnRate * (m_hat_w / (np.sqrt(v_hat_w) + epsilon))

    m_hat_b = (m_b) / (1-beta_1**t)
    v_hat_b = (v_b) / (1-beta_2**t)
    biasesChange = learnRate * (m_hat_b / (np.sqrt(v_hat_b) + epsilon))
    return weightsChange, biasesChange


class Adam:
    def __init__(self, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_w, self.v_w = 0, 0
        self.m_b, self.v_b = 0, 0
        self.m_k, self.v_k = 0, 0
    
    def optimizeFC(self, wGradients, bGradients, learnRate, t):
        #weightsChange, biasesChange = optimizeFCJIT(self.m_k, self.v_k, self.m_b, self.v_b, self.beta_1, self.beta_2, wGradients, bGradients, learnRate, self.epsilon, t)

        self.m_w = self.beta_1 * self.m_w + (1-self.beta_1) * wGradients
        self.v_w = self.beta_2 * self.v_w + (1-self.beta_2) * wGradients**2

        self.m_b = self.beta_1 * self.m_b + (1-self.beta_1) * bGradients
        self.v_b = self.beta_2 * self.v_b + (1-self.beta_2) * bGradients**2

        epsilon = 1e-8
        m_hat_w = (self.m_w) / (1-self.beta_1**t)
        v_hat_w = (self.v_w) / (1-self.beta_2**t)
        weightsChange = learnRate * (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon))

        m_hat_b = (self.m_b) / (1-self.beta_1**t)
        v_hat_b = (self.v_b) / (1-self.beta_2**t)
        biasesChange = learnRate * (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))

        return weightsChange, biasesChange
        

    def optimizeConv(self, kGradients, learnRate, t):
        #kernelsChange = optimizeConvJIT(self.m_k, self.v_k, self.beta_1, self.beta_2, kGradients, learnRate, self.epsilon, t, m_hat_k = 0, v_hat_k = 0)
        self.m_k = self.beta_1 * self.m_k + (1-self.beta_1) * kGradients
        self.v_k = self.beta_2 * self.v_k + (1-self.beta_2) * (kGradients**2)

        m_hat_k = (self.m_k) / (1-self.beta_1**t)
        v_hat_k = (self.v_k) / (1-self.beta_2**t)
        kernelsChange = learnRate * (m_hat_k / (np.sqrt(v_hat_k) + self.epsilon))
        return kernelsChange
