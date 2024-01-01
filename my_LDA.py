import numpy as np
import scipy.linalg as linalg

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    
    classLabels = np.unique(Y)
    classNum = len(classLabels)
    datanum, dim = X.shape
    totalMean = np.mean(X, 0)

    # partition class labels per label - list of arrays per label
    partition = [np.where(Y == label)[0] for label in classLabels]

    # find mean value per class (per attribute) - list of arrays per label
    classMean = [(np.mean(X[idx], axis=0), len(idx)) for idx in partition]

    # Compute the within-class scatter matrix
    Sw = np.zeros((dim,dim))
    for idx in partition:
        Sw += np.cov(X[idx], rowvar=0) * (len(idx)-1)# covariance matrix * fraction of instances per class

    # Compute the between-class scatter matrix
    Sb = np.zeros((dim,dim))
    for mu,class_size in classMean:
        temp = mu-totalMean
        temp = temp.reshape(dim,1)
        Sb += class_size * np.dot(temp, np.transpose(temp))    

    # Solve the eigenvalue problem for discriminant directions to maximize class seperability while simultaneously minimizing
    # the variance within each class
    # The exception code can be ignored for the example dataset
    try:
        S = np.dot(linalg.inv(Sw), Sb)
        eigval, eigvec = linalg.eig(S)
    except: #Singular Matrix
        print("Singular matrix")
        eigval, eigvec = linalg.eig(Sb, Sw+Sb)

    idx = eigval.argsort()[::-1] # Sort eigenvalues
    eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
    W = np.real(eigvec[:,:classNum-1]) # eigenvectors correspond to k-1 largest eigenvalues


    # Project data onto the new LDA space
    X_lda = np.real(np.dot(X, np.real(W)))

    # project the mean vectors of each class onto the LDA space
    projected_centroid = [np.dot(mu, np.real(W)) for mu,class_size in classMean]

    return W, projected_centroid, X_lda
