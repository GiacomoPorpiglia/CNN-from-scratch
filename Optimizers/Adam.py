import numpy as np

class Adam:
    def __init__(self, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_w, self.v_w = 0, 0
        self.m_b, self.v_b = 0, 0
        self.m_k, self.v_k = 0, 0
    
    def optimizeFC(self, wGradients, bGradients, learnRate, t):
    
        self.m_w = self.beta_1 * self.m_w + (1-self.beta_1) * wGradients
        self.v_w = self.beta_2 * self.v_w + (1-self.beta_2) * wGradients**2

        self.m_b = self.beta_1 * self.m_b + (1-self.beta_1) * bGradients
        self.v_b = self.beta_2 * self.v_b + (1-self.beta_2) * bGradients**2

        m_hat_w = (self.m_w) / (1-self.beta_1**t)
        v_hat_w = (self.v_w) / (1-self.beta_2**t)
        weightsChange = learnRate * (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon))

        m_hat_b = (self.m_b) / (1-self.beta_1**t)
        v_hat_b = (self.v_b) / (1-self.beta_2**t)
        biasesChange = learnRate * (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))
        
        return weightsChange, biasesChange
        

    def optimizeConv(self, kGradients, learnRate, t):  
        self.m_k = self.beta_1 * self.m_k + (1-self.beta_1) * kGradients
        self.v_k = self.beta_2 * self.v_k + (1-self.beta_2) * (kGradients**2)

        m_hat_k = (self.m_k) / (1-self.beta_1**t)
        v_hat_k = (self.v_k) / (1-self.beta_2**t)
        kernelsChange = learnRate * (m_hat_k / (np.sqrt(v_hat_k) + self.epsilon))
        return kernelsChange
