import numpy as np
from sklearn.datasets import make_blobs
from src.neuron import Neuron
import matplotlib.pyplot as mlt
import matplotlib.animation as animation

# 1. Generate linearly separable binary classification data
# X_raw: (100, 2) -> 100 samples, 2 features
# y_raw: (100,)   -> target labels 0 or 1
X_raw, y_raw = make_blobs(
    n_samples=100,
    n_features=2,
    centers=2,
    cluster_std=1.2,
    random_state=42
)

# 2. Transpose X to shape (num_features, num_samples) -> (2, 100)
X_train = X_raw.T
y_train = y_raw

# 3. Fit Neuron on 2-feature input
singleNeuron = Neuron(numInputs=2)
'''singleNeuron.fit(
    X_train, y_train,
    batchSize=16,
    epochs=100,
    learningRate=0.05
)'''

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

    ax[0].set_title("Decision Boundary")
    ax[0].set_xlabel("x_1")
    ax[0].set_ylabel("x_2")
    plotDataPoints = ax[0].scatter(inputs[0, :],inputs[1, :],c=outputs,cmap='bwr',alpha = 0.1) # all data points
    ax[0].set_xlim(np.min(inputs[0,:]-1),np.max(inputs[0,:]+1))
    ax[0].set_ylim(np.min(inputs[1,:]-1),np.max(inputs[1,:]+1))

    plotLine, = ax[0].plot([],[], 'k--',linewidth=2,label='Decision Boundary') 
                            # all the .plot outputs seem to come in the form [output], which causes a " 'list' has no attribute 'set_data' error"
                            # using "," allows us to only set the first entry (the only entry) to the variable, removing the list

    ax[1].set_title("Training Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlim(0,epochs)
    #ax[1].set_ylim(0,120_000)
    plotLoss, = ax[1].plot([],[],'b')


    ax[2].set_title("Parameters")
    ax[2].set_xlabel("Epochs")
    ax[2].set_xlim(0,epochs)
    #[2].set_ylim(-2,4)
    plotW1, = ax[2].plot([],[],'C5', label="w1")
    plotW2, = ax[2].plot([],[],'C6', label="w2")
    plotBiases, = ax[2].plot([],[],'C8', label="b")
    ax[2].legend()

    def init():
        return plotLine, plotLoss, plotW1, plotW2, plotBiases

    def update(epoch):
        loss, Wi,Bi = singleNeuron.singleTrainingLoop(inputs,outputs,batchSize=sizeOfBatch,learningRate=lr)

        Weight_vals.append(Wi.flatten())
        #print(Wi)
        bias_vals.append(Bi)
        loss_vals.append(loss)
        #print(loss)

        x1_vals = np.linspace(ax[0].get_xlim()[0], ax[0].get_xlim()[1], 200)
        w1, w2 = Weight_vals[-1][0], Weight_vals[-1][1]
        b = bias_vals[-1]

        if np.abs(w2) > 1e-5: #prevent division by zero 
            x2_vals = -(w1 * x1_vals + b)/w2
            plotLine.set_data(x1_vals,x2_vals)

        ax[1].set_ylim(0, max(loss_vals))
        epochsSoFar = list(range(1,len(loss_vals) + 1))
        plotLoss.set_data(epochsSoFar,loss_vals)

        w1History = [w[0] for w in Weight_vals]
        w2History = [w[1] for w in Weight_vals]

        plotW1.set_data(epochsSoFar, w1History)
        plotW2.set_data(epochsSoFar, w2History)

        plotBiases.set_data(epochsSoFar,bias_vals)

        allParams = w1History + w2History + bias_vals
        ax[2].set_ylim(min(allParams) - 0.5, max(allParams) + 0.5)

        return plotLine,plotLoss,plotW1,plotW2,plotBiases

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

