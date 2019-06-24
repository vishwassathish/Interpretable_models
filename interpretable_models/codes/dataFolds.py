import csv
import random
import numpy as np
from pickle import dump

def readData(filename):
    data=[]
    with open(filename,"r") as f:
        reader=csv.reader(f)
        for row in reader:
            data.append(row)
        headers=data[0]
        data=data[1:]
    f.close()
    for row in data:
        for i in range(1,len(row)):
            row[i]=float(row[i])
    return data,headers

def divideData(ratio,data):
    l=len(data)
    numberTraining=int(ratio*l)
    numberTesting=l-numberTraining
    dataPositive=[]
    dataNegative=[]
    for row in data:
        if(row[-1]==0.0):
            dataNegative.append(row)
        else:
            dataPositive.append(row)
    trainingData=[]
    testingData=[]
    numDataPositive=len(dataPositive)
    while(len(dataPositive)>=numDataPositive//2):
        n=random.randint(0,len(dataPositive)-1)
        testingData.append(dataPositive[n])
        del dataPositive[n]
    #print("Number Testing : ",len(testingData))
    remaining=numberTesting-len(testingData)
    for i in range(0,remaining):
        n=random.randint(0,len(dataNegative)-1)
        testingData.append(dataNegative[n])
        del dataNegative[n]
    trainingData=dataPositive+dataNegative
    #print("Number Training : ",len(trainingData))
    
    return trainingData,testingData

def getTrainAndTest(train,test):
    print("splitting x and y")
    xTrain=[]
    xTest=[]
    yTrain=[]
    yTest=[]
    for row in train:
        xTrain.append(np.array(row[0:len(row)-1]))
        yTrain.append(row[-1])
    for row in test:
        xTest.append(np.array(row[0:len(row)-1]))
        yTest.append(row[-1])
    for x in xTest:
        try:
            while(True):
                idx=xTrain.index(x)
                del xTrain[idx]
                del yTrain[idx]
        except:
            pass
    return np.array(xTrain).astype(float),np.array(yTrain).astype(float),np.array(xTest).astype(float),np.array(yTest).astype(float)



if __name__ == "__main__" :

    path='../data/final_training_data.csv'
    data,featureNames=readData(path)
    train,test=divideData(0.65,data)
    x_train, y_train, x_validate, y_validate=getTrainAndTest(train,test)
    
    print("Training data: ", x_train.shape, " ", y_train.shape)
    print("Testing data: ", x_validate.shape, " ", y_validate.shape)

    myData = {"feature_names":featureNames, "x_train":x_train, "y_train":y_train, "x_validate":x_validate, "y_validate":y_validate}
    with open("../data/myData.pickle", "wb") as f:
        dump(myData, f)
    # with open("../data/x_train.pickle", "wb") as f:
    #   dump(x_train, f)
    # with open("../data/y_train.pickle", "wb") as f:
    #   dump(y_train, f)
    # with open("../data/x_validate.pickle", "wb") as f:
    #   dump(x_validate, f)
    # with open("../data/y_vaidate.pickle", "wb") as f:
    #   dump(y_validate, f)