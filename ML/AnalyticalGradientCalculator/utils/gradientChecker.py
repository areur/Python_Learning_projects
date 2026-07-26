import numpy as np
from src.neuron import neuron
# apparently this file was the week 7 project - Autodiff Finite-Difference Engine
# Not sure where the AutoDiff part comes in.. perhaps I need to write better prompts
# when asking for projects

h = 10**-4 # small value used in limits

# dh = [dL/dw1, dL/dw2, ...] array of partial derivatives dL/dw, from the limit definition (pg. 126 of Mathematics for Machine Learning)
# dh_i = index i partial derivative (dL/dwi)
def checkGradient(x,weights,bias,y,calculated_weightLoss,calculated_biasLoss):
    dh_dw = np.empty_like(calculated_weightLoss,dtype=np.float64)

    temp_weights = weights.copy()

    # Finite-difference check for weights
    for i in range(len(weights)):
        
        temp_weights.flat[i] += h
        plus_h = neuron();
        
        loss_plus, _ = plus_h.forwardPass(x,temp_weights,bias,y)

        temp_weights[i] -= 2*h
        minus_h = neuron();
        loss_minus, _ = minus_h.forwardPass(x,temp_weights,bias,y)
        
        limit = (loss_plus - loss_minus)/(2*h)
        dh_dw.flat[i] = np.squeeze(limit)

        temp_weights.flat[i] += h
        # spamming flat to ensure I do not have nested arrays for no reason 
        #ex. [[1]]

    # Finite-difference check for bias
    plus_h = neuron();
    biasloss_plus, _ = plus_h.forwardPass(x,weights,bias+h,y)

    minus_h = neuron();
    biasloss_minus, _ = minus_h.forwardPass(x,weights,bias-h,y)
    
    dh_db = (biasloss_plus - biasloss_minus)/(2*h)

    # Combine weight and bias gradients so u can check 1 vector
    checkerGrad = np.concatenate([dh_dw.ravel(),[np.squeeze(dh_db)]])
    inputGrad = np.concatenate([calculated_weightLoss.ravel(),[np.squeeze(calculated_biasLoss)]])

    denom = np.linalg.norm(checkerGrad) + np.linalg.norm(inputGrad)
    numer = np.linalg.norm(checkerGrad - inputGrad) / denom
    error = numer/denom
    return error < (10**-6)