import numpy as np
from sklearn.svm import SVC
from numpy import genfromtxt
from sklearn.gaussian_process import GaussianProcess
import time
import ClassifierScore
import numpy

Train_features = genfromtxt('Train_features', dtype='float', delimiter=',')
Train_Labels = genfromtxt('Train_Labels', dtype='int', delimiter=',')
Test_features = genfromtxt('Test_features', dtype='float', delimiter=',')
Test_Labels = genfromtxt('Test_Labels', dtype='int', delimiter=',')

predicted_labels = numpy.zeros((len(Train_Labels.T), len(Train_Labels)));
#print(predicted_labels.size);
Total_Time = 0;

for i in range(len(Train_Labels)):

    clf = GaussianProcess(regr='constant', corr='squared_exponential', beta0=None, storage_mode='full',
                          verbose=True, theta0=0.1, thetaL=None, thetaU=None, optimizer='fmin_cobyla',
                          random_start=1, normalize=True, nugget=2.2204460492503131e-15, random_state=None)


    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:,i] = clf.predict(Test_features.T);
    Total_Time = Total_Time + ((time.time() - start_time));
    #print(" %s seconds " % (time.time() - start_time))

print(" %s seconds " % Total_Time)
acc, sen, spe = ClassifierScore.CalculateScore(Test_Labels.T, predicted_labels)
ClassifierScore.PrintScore(acc, sen, spe)