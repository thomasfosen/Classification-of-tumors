import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
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

# read data and remove incomplete rows
data = pd.read_csv('breast-cancer-wisconsin.data', header = None, names = headers)
data = data[data.BareNuclei != "?"]
array = np.asarray(data).astype('float32')

# normalize output variables
for i in range(len(array)):
    if array[i, -1] == 2:
        array[i, -1] = 0
    else:
        array[i, -1] = 1

# split to training and testing set

splitPerc = 0.8 #adjustable split percentage between training and test data
trainingData = array[:round(len(array)*splitPerc)]
testingData = array[round(len(array)*splitPerc):]



# create test and train tensors
xTrain =  tf.convert_to_tensor(trainingData[:, 1 : -1])
yTrain = tf.convert_to_tensor(trainingData[:,-1])

xTest = tf.convert_to_tensor(testingData[:, 1 : -1])
yTest = tf.convert_to_tensor(testingData[:, -1])


''' Creating model '''


# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(10, activation="relu", name="inputLayer"),
        layers.Dense(15, activation="relu", name="hiddenLayer"),
        layers.Dense(1, name="outputLayer"),
    ]
)

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]
)

history = model.fit(xTrain, yTrain, epochs = 15)


#make predictions

predictions = model.predict(xTest)

#turn into classes
prediction_classes = [
    1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
]

print(f'Accuracy: {accuracy_score(yTest, prediction_classes):.2f}')
