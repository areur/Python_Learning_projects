def squaredLoss(yhat,y):
    accumulatedLoss = 0.0
    for i in range(y.shape[0]):
        accumulatedLoss += (yhat[0,i]-y[i])**2
    loss = (0.5*y.shape[0])*accumulatedLoss
    derivative = yhat-y
    return (loss,derivative)

