import math
import numpy as np
import scipy as sp
#-------------------------------------------------------------------
def log(n):
    return math.log(n)
#-------------------------------------------------------------------
def exp(n):
    return math.exp(n)
#-------------------------------------------------------------------
class logistic:
    #******************************************************
    def __init__(self, parameters):
        self.parameters = parameters

        self.x = [[1, -0.851064496, 0.0949158] ,[1, -0.250313087, -1.044073795], [1, 1.101377584, 0.949157996]]
        # self.x = [[1, 60, 155], [1, 64, 135], [1, 73, 170] ]
        self.y = [0, 1, 1]

    #******************************************************
    ########## Feel Free to Add Helper Functions ##########
    #******************************************************
    def log_likelihood(self):
        ll = 0.0
        ##################### Please Fill Missing Lines Here #####################
        tmp = np.dot(self.y,np.transpose(self.x))
        
        tmp = np.dot(tmp,self.parameters)
        ll += tmp

        for i in range(0,len(self.parameters)):
            power = np.dot(np.transpose(self.x[i]),self.parameters)
            tmp += log( 1 + exp(power) )
        ll -= tmp

        print "Log Likelihood:"
        print ll
        return ll

    #******************************************************
    def gradients(self):
        gradients = np.zeros(len(self.y))
        ##################### Please Fill Missing Lines Here #####################
        
        for j in range(len(self.x[0])):
            for i in range(len(self.x)):
                gradients[j] += self.x[i][j]*self.y[i]
                eBx = np.dot(self.parameters,self.x[i])
                eBx = exp(eBx)
                gradients[j] -= self.x[i][j]*eBx/(1+eBx)
        print "Gradient:"
        print gradients

        return gradients

    #******************************************************
    def iterate(self):
        ##################### Please Fill Missing Lines Here #####################
        test = self.log_likelihood()
        grad = self.gradients()
        hess = self.hessian()
        self.parameters = self.parameters - np.dot(np.linalg.inv(hess),grad)
        print np.dot(np.linalg.inv(hess),grad)
        print "----Iter done parameters:"
        print self.parameters
        print ""

        return self.parameters

    #******************************************************
    def hessian(self):
        n = len(self.parameters)
        hessian = np.zeros((n, n))
        ##################### Please Fill Missing Lines Here #####################
 
        for j in range(0,n):
            for k in range(0,n): 
                for i in range(len(self.x)):
                    eBx = exp(np.dot(self.parameters,self.x[i]))
                    p = eBx/(1+eBx)
                    hessian[j][k] -= self.x[i][j]*self.x[i][k]*p*(1-p)
        
        print "Hessian:"
        print hessian
        
        return hessian

#-------------------------------------------------------------------
parameters = []
##################### Please Fill Missing Lines Here #####################

parameters = [0.25, 0.25, 0.25]
## initialize parameters
l = logistic(parameters)
parameters = l.iterate()
l = logistic(parameters)
parameters = l.iterate()
