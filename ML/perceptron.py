"""
A Perceptron is like the simplest version of a Neural Network
It takes multiple inputs
applies weights (which are shifted each training loop) and gets a weighted sum
adds a bias (constant that decides when the perceptron activates)
outputs a value that is put into an Activation Function (in our case a step function)
The Activation function returns a binary value based on the value of the output
error is calculated to help adjust weights & bias
"""
import numpy as np
import matplotlib.pyplot as mpl

x = np.array([[0,0], [1,0], [0,1], [1,1]]) # AND Gate
y = np.array([0,0,0,1]) # Output 

def stepFunction(input):
    return 1 if input > 0 else 0

class Perceptron:
    def __init__(self,learningRate=0.1, epochs=20):
        self.learnRate = learningRate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    def predict(self,xIn):
        z = np.dot(xIn, self.weights) + self.bias # Dot product
        return z
    def fit(self,x,y):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)
        for epoch in range(self.epochs):
            for i in range(len(x)):
                xi = x[i]
                yOut = y[i]
                z = self.predict(xi)
                yPred = stepFunction(z)

                error = yOut - yPred
                adjustment = self.learnRate*error
                self.weights += adjustment*xi
                self.bias += adjustment
                print("Attempt #",epoch,"Input:",xi,"Prediction:",yPred)
                print("Error:",error,"z",z)
                if i == 3: 
                    print("---")

def plotDecisionBoundary(x,y,model,title):
    xMin,xMax = x[:,0].min() -1, x[:,0].max() + 1
    yMin,yMax = x[:,1].min() -1, x[:,1].max() + 1

    xx,yy = np.meshgrid(
        np.linspace(xMin,xMax,300),
        np.linspace(yMin,yMax,300)
    )

    grid = np.c_[xx.ravel(),yy.ravel()]
    output = model.predict(grid)
    zz = np.where(output >= 0, 1, 0)
    zz = zz.reshape(xx.shape)
    print(np.unique(zz))
    mpl.figure(figsize=(6,5))
    mpl.contour(xx,yy,zz, alpha=0.3, cmap="GnBu")
    for label in np.unique(y):
        points = x[y == label]
        mpl.scatter(points[:,0], points[:,1],
                    s=100, edgecolor='black',
                    label=f"Class {label}")
        
    mpl.title(title)
    mpl.xlabel("x1")
    mpl.ylabel("y1")
    mpl.legend()
    mpl.grid(True)
    mpl.show()

predictAnd = Perceptron()
predictAnd.fit(x,y)
plotDecisionBoundary(x,y,predictAnd,"Perceptron Decision Boundary (AND)")
"""
weights = np.random.rand(2)
bias = np.random.rand(1)

learningRate = 0.1

#Training loop
for epoch in range(5):
    for i in range(len(x)):
        xIn = x[i]
        yOut = y[i]

        #Weighted sum
        z = np.dot(xIn, weights) + bias # Dot product
        yPredicted = step(z)

        #Update rule
        error = yOut - yPredicted
        weights += learningRate*error*xIn #modify weights
        bias += learningRate*error
        print("Attempt #",epoch,"Input:",xIn,"Prediction:",yPredicted)
        print("Error:",error,"z",z)
        if i == 3: 
            print("---")


print("Trained weights:",weights)
print("Trained bias:", bias)
"""