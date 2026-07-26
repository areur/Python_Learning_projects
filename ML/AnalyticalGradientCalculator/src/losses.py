def squaredLoss(yhat,y):
    loss = 0.5*(yhat-y)**2
    derivative = yhat-y
    return (loss,derivative)

