import numpy as np
import sys, time

def ManhattanLength(x):
    sum = 0
    for element in x:
        sum += np.abs(element) 
    return sum

def ManhattanDistance(x,y):
    sum = 0
    for index in x:
        sum += np.abs(x[index]-y[index])
    return sum

def EuclideanLength(x):
    sum = 0
    for element in x:
        sum += element^2
    return sum

def EuclideanDistance(x,y):
    sum = 0
    '''for index in x:
        sum += (x[index]-y[index])^2'''
    diff = x-y
    dotProd = np.dot(diff,diff)
    sum = np.sqrt(dotProd)
    return sum



def predictkNN(X_train,y_train,homeTestPoint,k,distanceMetric):
    '''
    k-Nearest Neighbors Classifiers determine the class of a single data point
    by looking at the classes of the k number of nearest points nearest to 
    the point of interest and predicting it to be the class most of the closest
    points are
    '''
    pointDistances = list()

    #Loop through all test points in the training data to populate the list
    for i in range(len(X_train)):
        currVector = X_train[i]

        if distanceMetric == "manhattan":
            dist = ManhattanDistance(homeTestPoint,currVector)
        elif distanceMetric == "euclidean":
            dist = EuclideanDistance(homeTestPoint,currVector)

        #Append dist label tuple to pointDistances
        pointDistances.append((dist,y_train[i]))
    
    nearestNeighbors = pointDistances[:k]
    # get only the labels (second in tuple)
    neighborLabels = [label for dist,label in nearestNeighbors]

    predictedLabel = max(set(neighborLabels),key=neighborLabels.count)
        #set removes duplicates
        #max finds the max value
            #Whatever "key=" is set to is what the max function finds the max of

    return predictedLabel

def main():
    X_train = np.array([
    [120, 8],   # Apple
    [130, 7],   # Apple
    [115, 9],   # Apple
    [30,  3],   # Lemon
    [40,  2],   # Lemon
    [35,  4],   # Lemon
    [150, 2],   # Watermelon
    [160, 3],   # Watermelon
    [155, 2]    # Watermelon
])
    y_train = np.array([
    "Apple", 
    "Apple", 
    "Apple", 
    "Lemon", 
    "Lemon", 
    "Lemon", 
    "Watermelon", 
    "Watermelon", 
    "Watermelon"
])

    testPoint = np.array([122, 8])
    kNeighbors = 3

    cont = True
    while (cont == True):
        choice = input("[1] Predict w/ Manhattan Distance\n[2] Predict w/ Euclidean\n[3] Exit Program\n")
        match choice:
            case "1":
                prediction = predictkNN(X_train,y_train,testPoint,kNeighbors,"manhattan")
                print(prediction)
            case "2":
                prediction = predictkNN(X_train,y_train,testPoint,kNeighbors,"euclidean")
                print(prediction)
            case "3":
                break
            case _: 
                continue

if __name__ == "__main__":
    main()
