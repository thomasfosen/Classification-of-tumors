import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


'''Data Config'''

headers = [ \
    'Sample Code Number', \
    'Clump Thickness', \
    'Uniformity of Cell Size', \
    'Uniformity of Cell Shape', \
    'Marginal Adhesion', \
    'Single Epithelial Cell Size', \
    'BareNuclei', \
    'Bland Chromatin', \
    'Normal Nucleoli', \
    'Mitoses', \
    'Class']

# read data, assign headers and remove incomplete rows
data = pd.read_csv('breast-cancer-wisconsin.data', header = None, names = headers)
data = data[data.BareNuclei != "?"] # Creates new data without compromised rows
array = np.asarray(data).astype('float32')

# normalize output variables --> 1 indicates malignant, 0 indicates benign
for i in range(len(array)):
    if array[i, -1] == 2:
        array[i, -1] = 0
    else:
        array[i, -1] = 1

# split to training and testing set
splitPercentage = 0.8 #adjustable split percentage between training and test data
trainingData = array[:round(len(array)*splitPercentage)]
testingData = array[round(len(array)*splitPercentage):]


# create test and train tensors
xTrain =  tf.convert_to_tensor(trainingData[:, 1 : -1])
yTrain = tf.convert_to_tensor(trainingData[:,-1])
xTest = tf.convert_to_tensor(testingData[:, 1 : -1])
yTest = tf.convert_to_tensor(testingData[:, -1])


''' Creating model '''


# Initiate model
model = keras.Sequential(
    [
        layers.Dense(10, activation = "relu", name = "inputLayer"),
        layers.Dense(20, activation = "relu", name = "hiddenLayer"),
        layers.Dense(1, name = "outputLayer"),
    ]
)

# Compile model
model.compile(
    loss = tf.keras.losses.binary_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]
)


fitModel = model.fit(xTrain, yTrain, epochs = 15)
classificationValues = model.predict(xTest)


# Check and calculate final accuracy
for i in range(len(classificationValues)):
    if classificationValues[i] < 0.5:
        classificationValues[i] = 0
    else:
        classificationValues[i] = 1

results = open('results2.txt', 'w')
results.write(str(classificationValues))
results.close()

wrong = 0
right = 0

for i in range(len(classificationValues)):
    if classificationValues[i] == testingData[i, -1]:
        right += 1
    else:
        wrong += 1

accuracy = right / (wrong + right)

print('Right predictions: ' + str(right))
print('Wrong predictions: ' + str(wrong))
print("Final accuracy: " + str(accuracy))

model.save('model2')
