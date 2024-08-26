class Momentum:
    def __init__(self, momentum):
        self.momentum = momentum
        self.w_change, self.b_change = 0, 0
        self.k_change = 0
    
    def optimizeFC(self, wGradients, bGradients, learnRate):
        self.w_change = wGradients*learnRate + self.w_change*self.momentum
        self.b_change = bGradients*learnRate + self.b_change*self.momentum
        
        return self.w_change, self.b_change
        

    def optimizeConv(self, kGradients, learnRate):
        self.k_change = kGradients*learnRate + self.k_change* self.momentum
        return self.k_change