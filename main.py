import numpy as np
from my_LDA import my_LDA
from predict import predict

# Load data (Wine dataset)
np.random.seed(1)
my_data = np.genfromtxt('wine_data.csv', delimiter=',')
np.random.shuffle(my_data)  # shuffle datataset

n_train = 100
trainingData = my_data[:n_train, 1:]  # training data
trainingLabels = my_data[:n_train, 0]  # class labels of training data

testData = my_data[n_train:, 1:]  # training data
testLabels = my_data[n_train:, 0]  # class labels of training data


# training LDA classifier
W, projected_centroid, X_lda = my_LDA(trainingData, trainingLabels)

# Perform predictions for the test data
predictedLabels = predict(testData, projected_centroid, W)
predictedLabels = predictedLabels+1


# Compute accuracy
counter = 0
for i in range(predictedLabels.size):
    if predictedLabels[i] == testLabels[i]:
        counter += 1
print('Accuracy of LDA: %f' % (counter / float(predictedLabels.size) * 100.0))

