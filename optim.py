import numpy as np
from math import e, log, sqrt

#Exponential Decay Function
def EDL(time_period,start_lr,decay_rate=0.001):
    return start_lr*e**(-decay_rate*time_period)

#Adam optimizer
class Adam:
    def __name__(self):
        return "Adam"
    def __init__(self,lr=0.0001,beta1=0.9,beta2=.999):
        self.lr = lr
        self.t=1
        #weights
        self.m = 0
        self.v = 0
        self.mhat = 0
        self.vhat = 0
        self.alpha = lr
        self.beta1 = beta1
        self.beta2 = beta2
        #biases
        self.bm = 0
        self.bv = 0
        self.bmhat = 0
        self.bvhat = 0
        self.bbeta1 = beta1
        self.bbeta2 = beta2
    
    #will take in a gradient and return ADAM
    def adam(self,grad,bgrad):
        #these will be multiplied by the gradients in the update step
        #weights
        self.beta1 = self.beta1**self.t
        self.beta2 = self.beta2**self.t
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        self.mhat = self.m/(1-self.beta1)
        self.vhat = self.v/(1-self.beta2)
        change = self.lr * self.mhat / (np.sqrt(self.vhat) + 1.0e-8)
        
        
        #biases
        self.bbeta1 = self.bbeta1**self.t
        self.bbeta2 = self.bbeta2**self.t
        self.bm = self.beta1*self.bm + (1-self.bbeta1)*bgrad
        self.bv = self.beta2*self.bv + (1-self.bbeta2)*(bgrad**2)
        self.bmhat = self.bm/(1-self.bbeta1)
        self.bvhat = self.bv/(1-self.bbeta2)
        bchange = self.lr * self.bmhat / (np.sqrt(self.bvhat) + 1.0e-8)
        
        self.t += 1
        return change, bchange