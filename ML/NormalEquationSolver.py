import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

"""
-Modification for handling intercepts
    -weights[0] should be the intercept
    -Feature matrix X should have its first column
     be 1s as this will be multipied by weights[0]
    

-Method to calculate the weight vector (lambda)
    -Verify compatible dimensions for matrix mult 

-Store weights in the class
    -weights should be accessible

-Prediction function (kinda like kaggle)
    -uses dot product. y = Xw

-Cross checker with scikit learn LinearRegression
    -Should have a tolerance value

Optional: Matplotlib UI
"""
class projectionRegression(BaseEstimator, RegressorMixin):

    def __init__(self, interceptEnabled=False):
        self.interceptEnabled = interceptEnabled

    def fit(self,X,y) -> projectionRegression:
        # Project y onto the space whos basis is X

        # Step 1: Ensure y is two dimensional
        y = np.asarray(y).squeeze()

        # Step 2: Check for intercepts and adjust
        Xhat = self.__interceptHandling(X)

        # Step 3: Find part of equation that gets inversed X^top * X
        Xtop = np.transpose(Xhat)
        toInverse = np.dot(Xtop,Xhat)
        Inverse = np.linalg.inv(toInverse)

        # Step 4: apply equation 
        self.weights_ = np.dot(np.dot(Inverse,Xtop),y)
        return self

    def predict(self,X_new): # multiply using the equation y = Xw 
        Xhat_new = self.__interceptHandling(X_new)
        res = np.dot(Xhat_new,self.weights_)
        return res

    def __interceptHandling(self,X):
        onesColumn = np.ones((X.shape[0],1))
        if (self.interceptEnabled):
            Xhat = np.column_stack((onesColumn,X))
        else:
            Xhat = X
        return Xhat
    
def main():
    model = projectionRegression(True)
    professional = LinearRegression()

    startingX = np.array([[2,4],
                        [5,3],
                        [9,7]])
    startingY = np.array([2,
                        5,
                        8])

    testingX = np.array([[6,7],
                        [2,1],
                        [1,7]])

    model.fit(startingX,startingY)
    professional.fit(startingX,startingY)

    projGuess = model.predict(testingX)
    LinRegGuess = professional.predict(testingX)

    print("projectionRegression:",projGuess)
    print("LinearRegression:",LinRegGuess)

if __name__ == "__main__":
    # finally figured out why people do this, 
    # i tried to import the class from this file and it ran the example
    main()

