import numpy as np
import scipy.linalg as linalg


def predict(X, projected_centroid, W):
    """Apply the trained LDA classifier on the test data 
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """

    # Project test data onto the LDA space defined by W 
    projected_data  = np.dot(X, W)
    
    # Compute distances from centroid vectors
    dist = [linalg.norm(data-centroid) for data in projected_data for centroid in projected_centroid]
    Y_raw = np.reshape(np.array(dist), (len(X),len(projected_centroid)))
    
    # Assign the label of the closed centroid vector
    label = Y_raw.argmin(axis=1)

    return label
