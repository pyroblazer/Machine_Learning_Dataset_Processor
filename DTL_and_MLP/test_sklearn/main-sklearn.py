import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier # neural network
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("iris.csv")
target = data[['species']].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])
newData = data.drop('species', axis = 1)


clf1 = MLPClassifier(solver='sgd', batch_size=32, alpha=0, momentum=0, nesterovs_momentum=False, activation='logistic', max_iter=400, hidden_layer_sizes=(100))
clf1.fit(newData,target)

print("Model (weights)")
print(clf1.coefs_)

df = pd.concat([newData, target], axis=1)
train, test = train_test_split(df, test_size=0.3)
trainX = train[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
trainY = train.species #output of our training data
testX = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
testY = test.species   #output value of test data

trainX.head(5)
trainY.head(5)

testX.head(5)
testY.head(5)



clf2 = MLPClassifier(solver='sgd', batch_size=32, alpha=0, momentum=0, nesterovs_momentum=False, activation='logistic', max_iter=400, hidden_layer_sizes=(100))
clf2.fit(trainX, trainY)

prediction = clf2.predict(testX)

print('The accuracy of the Multi-layer Perceptron is:',metrics.accuracy_score(prediction,testY))