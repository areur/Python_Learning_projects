import numpy as np

def reLU(z):
    answer = np.maximum(0,z) # takes the max value for each entry between 0 and z
                             # effectively eliminating all negative values
    derivative = np.heaviside(z,0) #step function w/ height of 1
                                   #and a value of 0 at z=0
    return (answer,derivative)                        

def sigmoid(z):
    answer = 1/(1+np.exp(-z))
    derivative = answer*(1-answer)  # this is equal to the actual derivative
                                    # --> np.exp(-z)/((1+np.exp(-z))**2)
                                    # the proof is left as an exercise for the reader
    return (answer,derivative) 