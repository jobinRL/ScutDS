

from skmultiflow.trees import HoeffdingTree
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import OzaBagging
import scutds
import numpy as np
import arff
import os
import time



os.chdir('C:/Users/Gabri/Documents/WinterYear4/scikit-multiflow-master')
def runScut(filein, fileout):
    #1 read the data with arff.
    data  = arff.load(open(filein))
    #TODO set timer
    #2: read the classes from the arff file.
    classes1 = [int(x) for x in data["attributes"][-1][1]]
    classes_recall = {}
    for value in classes1:
        classes_recall[int(value)] = []
        classes_recall[int(value)].append(0)
        classes_recall[int(value)].append(0)

    scut = scutds.ScutDS(classes1,OzaBagging(HoeffdingTree()))
    #3 run 1 1000 batch.
    #3.1 getting the 1000 first elements with the 1000 first class in 2 different lists.
    i = 0

    Xtmp = data["data"][i:i+1000]
    X = []
    y = []
    for var in Xtmp:
        X.append(var[:-1])
        y.append(int(var[-1]))

    scut.partial_fit(X,y,classes1)
    i += 1000

    while(i+1000 < len(data["data"])):
        print(len(data["data"]) - i)
        Xtmp = data["data"][i:i+1000]
        X = []
        y = []
        for var in Xtmp:
            X.append(var[:-1])
            y.append(int(var[-1]))

        result = scut.predict(X)

        for j in range(0,len(result)):
            if( y[j] == result[j]):
                classes_recall[y[j]][0] += 1
            else:
                classes_recall[y[j]][1] += 1

        print("prediction done")
        scut.partial_fit(X,y)
        print("fit done")
        i += 1000
    if( i < len(data)):
        Xtmp = data["data"][i:len(data["data"])]
        X = []
        y = []
        for var in Xtmp:
            X.append(var[:-1])
            y.append(int(var[-1]))
        result = scut.predict(X)

        for j in range(0,len(result)):
            if( y[j] == result[j]):
                classes_recall[y[j]][0] += 1
            else:
                classes_recall[y[j]][1] += 1
    #TODO retrieve time
    file = open(fileout,"w")
    for key,var in classes_recall.items():
        file.write(str(key) + "\n")
        file.write(str(var[0]/(var[0] + var[1])) + "\n")


runScut('Cloudy_Fog_Snow.arff','Cloudy_Fog_Snow.csv')
runScut('Clear_Fog_Ice.arff','Clear_Fog_Ice.csv')
runScut('Cloudy_Clear_Fog.arff','Cloudy_Clear_Fog.csv')
runScut('Cloudy_Clear_Ice.arff','Cloudy_Clear_Ice.csv')
runScut('Cloudy_Clear_Snow.arff','Cloudy_Clear_Snow.csv')
runScut('Cloudy_Clear_Snow_Ice.arff','Cloudy_Clear_Snow_Ice.csv')
runScut('Cloudy_Clear_Snow_Ice_Fog.arff','Cloudy_Clear_Snow_Ice_Fog.csv')
runScut('Cloudy_Snow_Ice.arff','Cloudy_Snow_Ice.csv')
runScut('Cloudy_Snow_Ice_Fog.arff','Cloudy_Snow_Ice_Fog.csv')
