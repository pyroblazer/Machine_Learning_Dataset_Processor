import pandas as pd
import numpy as np
from MultilayerPerceptron import *
from sklearn.model_selection import train_test_split

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 

# Kfold
from sklearn.model_selection import KFold

def experiment(mlp, filename):
	mlp.save_model_to_file(filename)
	mlp.load_model_from_file(filename)

def split():
	print("Split 90-10")

	data = pd.read_csv("iris.csv")
	inputs = data.drop('species', axis = 1)
	target = data['species']

	#training
	df = pd.concat([inputs, target], axis=1)
	train, test = train_test_split(df, test_size=0.1)

	trainX = train[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
	trainY = train.species # output of our training data
	testX = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
	testY = test.species   # output value of test data

	trainX = trainX.reset_index(drop=True)
	trainY = trainY.reset_index(drop=True)
	testX = testX.reset_index(drop=True)
	testY = testY.reset_index(drop=True)
	mlp = myMLP()
	mlp.fit(trainX, trainY)
	experiment(mlp, "Output-MLP-Split-Test.txt")
	prediction = mlp.predict(testX)
	confusion_matrix_results = confusion_matrix(testY.values, prediction)

	print("Confusion Matrix: ")
	print(confusion_matrix_results)
	print("Accuracy Score: ")
	print(accuracy_score(testY.values, prediction))
	print("Classification Report: ")
	print(classification_report(testY.values, prediction))



def kfold():
	print("K-Fold")

	#Training k-fold
	data = pd.read_csv("iris.csv")
	inputs = data.drop('species', axis = 1)
	target = data['species']
	df = pd.concat([inputs, target], axis=1)
	train, test = train_test_split(df, test_size=0.2)  # divide 150 data to x and y = 150-x

	inputX = train[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
	inputY = train.species # output of our training data
	testX = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
	testY = test.species   # output value of test data

	inputX = inputX.reset_index(drop=True) 
	inputY = inputY.reset_index(drop=True)
	testX = testX.reset_index(drop=True)
	testY = testY.reset_index(drop=True)

	#k-fold
	kf = KFold(n_splits=10)
	kf.get_n_splits(inputX)
	indices = kf.split(inputX)

	mainmlp = myMLP()
	maxacc = 0

	for train_index, test_index in indices: 
		trainX, valX = inputX.iloc[train_index], inputX.iloc[test_index]
		trainY, valY = inputY.iloc[train_index], inputY.iloc[test_index]

		trainX = trainX.reset_index(drop=True) 
		trainY = trainY.reset_index(drop=True)
		valX = valX.reset_index(drop=True)
		valY = valY.reset_index(drop=True)

		mlp = myMLP()
		mlp.fit(trainX, trainY)
		experiment(mlp, "Output-MLP-Kfold.txt")
		prediction = mlp.predict(valX)
		if (accuracy_score(valY.values, prediction) > maxacc):
			mainmlp = mlp
			maxacc = accuracy_score(valY.values, prediction)
	finalPrediction = mainmlp.predict(testX)
	print("Accuracy Score: ")
	print(accuracy_score(testY.values, finalPrediction))


split()
kfold()