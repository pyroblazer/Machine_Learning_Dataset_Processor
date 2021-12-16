import pandas as pd
from myc45 import *

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

# Kfold
from sklearn.model_selection import KFold

pd.options.mode.chained_assignment = None

def experiment(c45):
    c45.id3.tree_.save_to_file("Output-C45.txt")
    c45.id3.tree_.load_from_file("Output-C45.txt")
    dict = {
        "sepal_length": 5.5,
        "sepal_width": 3.3,
        "petal_length": 2.0,
        "petal_width": 0.65
    }
    print(c45.classify(c45.id3.tree_, dict))

def split():
    print("Split 90-10")

    df = pd.read_csv('iris.csv', sep=',')
    train, test = train_test_split(df, test_size=0.1)

    # Main Program Goes Here:
    data = pd.read_csv("iris.csv")

    inputs = data.drop('species', axis = 1)
    target = data['species']

    #training
    df = pd.concat([inputs, target], axis=1)
    train, test = train_test_split(df, test_size=0.1)
    train = train.reset_index(drop=True)

    testX = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
    testY = test.species   # output value of test data

    testX = testX.reset_index(drop=True)
    testY = testY.reset_index(drop=True)

    attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'species'

    c45 = myC45(train, target, attributes)
    prediction = []
    for i in range(len(testX)):
        prediction.append(c45.classify(c45.id3.tree_, testX.iloc[i]))
    confusion_matrix_results = confusion_matrix(testY.values, prediction)

    print("Confusion Matrix: ")
    print(confusion_matrix_results)
    print("Accuracy Score: ")
    print(accuracy_score(testY.values, prediction))
    print("Classification Report: ")
    print(classification_report(testY.values, prediction))
    # experiment(c45)


def kfold():
    print("K-Fold")
    #Training k-fold
    df = pd.read_csv('iris.csv', sep=',')
    data, test = train_test_split(df, test_size=0.2)  # divide 150 data to x and y = 150-x

    dataX = data[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
    dataY = data.species # output of our training data
    testX = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
    testY = test.species   # output value of test data

    dataX = dataX.reset_index(drop=True) 
    dataY = dataY.reset_index(drop=True)
    testX = testX.reset_index(drop=True)
    testY = testY.reset_index(drop=True)

    #k-fold
    kf = KFold(n_splits=10)
    kf.get_n_splits(data)
    indices = kf.split(data)

    mainc45 = None
    maxacc = 0;

    for train_index, test_index in indices: 
        train = data.iloc[train_index]
        train = train.reset_index(drop=True)

        valX, valY = dataX.iloc[test_index], dataY.iloc[test_index]
        valX = valX.reset_index(drop=True)
        valY = valY.reset_index(drop=True)

        attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target = 'species'
        c45 = myC45(train, target, attributes)
        prediction = []
        for i in range(len(valX)):
            prediction.append(c45.classify(c45.id3.tree_, valX.iloc[i]))
        if (accuracy_score(valY.values, prediction) > maxacc):
            mainc45 = c45
            maxacc = accuracy_score(valY.values, prediction)
    finalPrediction = []
    for i in range(len(testX)):
        finalPrediction.append(mainc45.classify(mainc45.id3.tree_, testX.iloc[i]))
    print("Accuracy Score: ")
    print(accuracy_score(testY.values, finalPrediction))
    # experiment(mainc45)


split()
kfold()
