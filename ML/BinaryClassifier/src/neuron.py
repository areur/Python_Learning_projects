import numpy as np
from numpy import ndarray
from src import activations
from src import losses

class Neuron:

    def __init__(self,numInputs:int = 1) -> None:
        #internal storage
        #Xavier Weight Initialization
        limit = np.sqrt(6.0/(numInputs+1)) #sqrt(6+(numInputs+numOutputs))

        self.weights = np.random.uniform(-limit, limit, size=(numInputs,1))
        self.bias = 0.1 # np.random.rand(1)

        #cache
        self.x: ndarray = np.empty([numInputs,1])
        self.dyhat_dz: ndarray | float = 0.0
        self.dL_dyhat: ndarray | float = 0.0

    def forwardPass(self, x: ndarray, y = None):
        # Calculate a neuron output from input, weight, and bias
        # Calculate how much the weights need to change, if a true prediction (y) was given

        self.x = x

        z = np.dot(self.weights.T, x) + self.bias
        yhat, self.dyhat_dz = activations.sigmoid(z)

        if y is not None: # y is not provided during inference
            loss, self.dL_dyhat = losses.squaredLoss(yhat,y)
            return (loss, yhat)
            

        return (None,yhat)

    def computeGradients(self):
        batchSize = self.x.shape[1] # shape is tuple --> (features,number of data points)

        # Calculate the gradient used to adjust weights during backpropagation
        derivs: ndarray = np.atleast_2d(np.asarray(self.dL_dyhat * self.dyhat_dz))

        # (3,1) = (1,1) * (3*1)
        dL_dw = np.dot(self.x,derivs.T) /batchSize # averaged over batch size
        dL_db = float(np.mean(derivs))

        return (dL_dw, dL_db)

    def getBatch(self, inputData: ndarray, targetData: ndarray, batchSize):
            numDataPoints = inputData.shape[1] # shape is tuple --> (features,number of data points)
            # create empty list of the indices in the training data
            indices = np.arange(numDataPoints)
            # shuffle dat list
            np.random.shuffle(indices)
            # get the shuffled versions of input and output
            xShuffled = inputData[:, indices]
            yShuffled = targetData[indices]
    
            numBatches = int(np.ceil(numDataPoints/batchSize))
            return (numDataPoints, xShuffled, yShuffled, numBatches)
    
    def fit(self,inputData: ndarray,targetData: ndarray,batchSize,epochs, learningRate=0.5):
        # Future: add validation_data
        for epoch in range(epochs):
            # PSUEDO-CODE
                #forwardPass
                #compute gradients
                #Adjust weights and biases negative direction of gradients
                #print epoch 
            numDataPoints, xShuffled, yShuffled,numBatches = self.getBatch(inputData,targetData,batchSize)
            epochLoss = 0.0

            for batch in range(numBatches):
                startIndex = batch*batchSize
                #smaller number between the last index and the batch size * batch+1)
                endIndex = min(startIndex+batchSize,numDataPoints)

                #get batches
                xBatch = xShuffled[:,startIndex:endIndex] # first dimension is the number of features, second is data points
                yBatch = yShuffled[startIndex:endIndex]

                #do the forward pass and gradient calculation
                loss, _ = self.forwardPass(xBatch,yBatch)
                dL_dw, dL_db = self.computeGradients()

                #adjust for gradient, optimize!
                self.weights -= (learningRate*dL_dw)
                self.bias -= (learningRate*dL_db)

                epochLoss += float(np.mean(loss)) if loss is not None else 0.0
            print(f"Attempt # {epoch+1}/{epochs}, Loss:{epochLoss/numBatches}")

    def singleTrainingLoop(self, inputData: ndarray, targetData: ndarray , batchSize,learningRate=0.5):
        # PSUEDO CODE
            # select a batch to work on if batchSize is not nil
            # Perform a single forwardPass --> Get loss gradients
            # Return gradients, parameters,
        numDataPoints, xShuffled, yShuffled,numBatches = self.getBatch(inputData,targetData,batchSize)
        loopLoss = 0.0

        weights, biases = [], []
        losses_weight, losses_bias = [], []

        weights.append(self.weights)
        biases.append(self.bias)

        for batch in range(numBatches):
            startIndex = batch*batchSize
            #smaller number between the last index and the batch size * batch+1)
            endIndex = min(startIndex+batchSize,numDataPoints)

            #get batches
            xBatch = xShuffled[:,startIndex:endIndex] # first dimension is the number of features, second is data points
            yBatch = yShuffled[startIndex:endIndex]

            #do the forward pass and gradient calculation
            loss, _ = self.forwardPass(xBatch,yBatch)
            dL_dw, dL_db = self.computeGradients()

            #adjust for gradient, optimize!
            self.weights -= (learningRate*dL_dw)
            self.bias -= (learningRate*dL_db)

            loopLoss += float(np.mean(loss)) if loss is not None else 0.0

            weights.append(self.weights)
            biases.append(self.bias)
            losses_weight.append(dL_dw)
            losses_bias.append(dL_db)

        return (loopLoss,self.weights,self.bias)

