import numpy as np
from sklearn.datasets import make_regression
from src.neuron import Neuron
import matplotlib.pyplot as mlt
import matplotlib.animation as animation

# Set fixed seed for reproducibility across test runs
np.random.seed(2)

# 1. Generate dataset: X shape (num_samples, num_inputs), y shape (num_samples,)
X_raw, y_raw = make_regression(
    n_samples=100, 
    n_features=1, 
    noise=1.0, 
    bias=2.0, 
    random_state=42
)

# transpose to fit class requirements (num_inputs, num_samples)
X_train = X_raw.T 

# 3. Apply ReLU-like thresholding to targets if testing ReLU activations
y_train = np.maximum(0, y_raw)

singleNeuron = Neuron(1)
""" singleNeuron.fit(
    X_train,y_raw,
    batchSize=16,
    epochs=100,
    learningRate=0.05
    )"""



# Graphing (PER EPOCH)
# PSUEDO CODE
#   Plot 1: predicted line vs data points
#       Data points red, line black
#       Collect current weights, biases per epoch to animate line
#       Data points used in the current batch will be a darker red than ones not
#        
#   Plot 2: Training Loss
#       Draw the training loss values over time
#   Plot 3: Parameters
#       Draw the Weight and Bias values over time
#   Important function: update, adds to the graphs each loop

def animate_sgd(inputs,outputs,sizeOfBatch,epochs,lr=0.05):
    Weight_vals, bias_vals = [], []
    loss_vals = []

    fig, ax = mlt.subplots(1,3)
    fig.set_size_inches(20,5)

    ax[0].set_title("Fitted Line")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    #ax[0].setxlim(..,..)
    #ax[0].setylim(..,..)
    plotDataPoints = ax[0].plot(inputs.T,outputs,'r.',alpha = 0.1) # all data points
    # plotBatches = ax[0].plot([],[], 'C3.') # points used in the batch
    plotLine, = ax[0].plot([],[], 'k') # all the .plot outputs seem to come in the form [output], which causes a " 'list' has no attribute 'set_data' error"
                                       # using "," allows us to only set the first entry (the only entry) to the variable, removing the list

    ax[1].set_title("Training Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlim(0,epochs)
    ax[1].set_ylim(0,120_000)
    plotLoss, = ax[1].plot([],[],'b')


    ax[2].set_title("Weights")
    ax[2].set_xlabel("Epochs")
    ax[2].set_xlim(0,epochs)
    ax[2].set_ylim(-2,4)
    plotWeights, = ax[2].plot([],[],'C5', label="W")
    plotBiases, = ax[2].plot([],[],'C8', label="b")
    ax[2].legend()

    def init():
        return [plotDataPoints]

    def update(epoch):
        loss, Wi,Bi = singleNeuron.singleTrainingLoop(inputs,outputs,batchSize=sizeOfBatch,learningRate=lr)

        Weight_vals.append(Wi)
        bias_vals.append(Bi)
        loss_vals.append(loss)
        print(loss)

        # dynamically size y-axis for loss
        ax[1].set_ylim(0,max(loss_vals))

        new_inputs = np.linspace(-10.0,10.5,300)
        plotLine.set_data(new_inputs, Weight_vals[-1]*new_inputs + bias_vals[-1])
        plotLoss.set_data(range(epoch),loss_vals)
        plotWeights.set_data(range(epoch),Weight_vals)
        plotBiases.set_data(range(epoch),bias_vals)
        return plotLine,plotLoss,plotWeights,plotBiases

    anim = animation.FuncAnimation(
        fig,
        update,
        frames= range(1,epochs+1),
        init_func=init,
        blit = False,
        interval = 100,
        repeat=False
    )
    return anim


anim = animate_sgd(X_train,y_raw,16,epochs=100)
mlt.show()

