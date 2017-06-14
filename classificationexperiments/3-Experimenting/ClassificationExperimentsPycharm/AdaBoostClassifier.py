from numpy import genfromtxt
from sklearn.ensemble import AdaBoostClassifier
import ClassifierScore
import time
import numpy


Train_features = genfromtxt('Train_features', dtype='float', delimiter=',')
Train_Labels = genfromtxt('Train_Labels', dtype='int', delimiter=',')
Test_features = genfromtxt('Test_features', dtype='float', delimiter=',')
Test_Labels = genfromtxt('Test_Labels', dtype='int', delimiter=',')


predicted_labels = numpy.zeros((len(Train_Labels.T), len(Train_Labels)));
#print(predicted_labels.size);
Total_Time = 0;

for i in range(len(Train_Labels)):

    clf = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R',
                             random_state=None);

    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:, i] = clf.predict(Test_features.T);
    Total_Time = Total_Time + ((time.time() - start_time));
    # print(" %s seconds " % (time.time() - start_time))

print(" %s seconds " % Total_Time)
acc, sen, spe = ClassifierScore.CalculateScore(Test_Labels.T, predicted_labels)
ClassifierScore.PrintScore(acc, sen, spe)

#savetxt('tested_it.csv', predicted_labels.T, delimiter=",")