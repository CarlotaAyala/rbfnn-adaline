import numpy as np
import pandas as pd

def getInitialCentroids(k, data):
    minimum = data.min().min()
    maximum = data.max().max()
    centroids = list()
    for _ in range(k):
        centroid = np.random.uniform(minimum, maximum, data.shape[1])
        centroids.append(centroid)
    centroids = pd.DataFrame(centroids, columns=data.columns)

    return centroids

def sumSquared(vector1, vector2):
    return np.sum((vector1 - vector2) ** 2)

def calculatelError(vector1, vector2):
    return np.sqrt(sumSquared(vector1, vector2))

def calculateErrorForAllCentroids(data, sampleIndex, centroids):
    errors = np.array([])
    for centroidIndex in range(centroids.shape[0]):
        error = calculatelError(centroids.iloc[centroidIndex, :2], data.iloc[sampleIndex, :2])
        errors = np.append(errors, error)
    return errors

def assignCentroid(data, centroids):
    assignedCentroid = list()
    centroidErrors = list()
    for sampleIndex in range(data.shape[0]):
        errors = calculateErrorForAllCentroids(data, sampleIndex, centroids)
        closestCentroid = np.where(errors == np.amin(errors))[0].tolist()[0]
        centroidError = np.amin(errors)
        assignedCentroid.append(closestCentroid)
        centroidErrors.append(centroidError)
    return assignedCentroid, centroidErrors

def kmeans(data, k):
    while True:
        centroids = getInitialCentroids(k, data)
        data['centroid'], _ = assignCentroid(data, centroids)
        if np.unique(data['centroid']).__len__() == k:
            break
    error = list()
    index = 0
    while True:
        data['centroid'], errorIteration = assignCentroid(data, centroids)
        error.append(sum(errorIteration))
        centroids = data.groupby('centroid').agg('mean').reset_index(drop=True)
        if len(error) >= 2:
            if round(error[index], 3) == round(error[index - 1], 3):
                break
        index += 1
    data['centroid'], errorIteration = assignCentroid(data, centroids)
    centroids = data.groupby('centroid').agg('mean').reset_index(drop=True)
    return data['centroid'], errorIteration, centroids