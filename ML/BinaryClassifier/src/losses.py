import numpy as np

def squaredLoss(yhat, y):
    # Flatten inputs to 1D to avoid implicit broadcasting issues
    yhat_flat = np.asarray(yhat).squeeze()
    y_flat = np.asarray(y).squeeze()
    
    N = y_flat.shape[0]
    loss = (1 / N) * np.sum((yhat_flat - y_flat) ** 2)
    derivative = (2/N) * (yhat - y)
    
    return (loss, derivative)

def binary_CrossEntropyLoss(yhat, y):
    accumulatedLoss = 0.0
    N = y.shape[0]
    loss = -(1/N)*np.sum(y * np.log(yhat) + (1-y) * np.log(1-yhat))
    derivative = -((y/yhat) - ((1-y)/(1-yhat)))
    return (loss, derivative)