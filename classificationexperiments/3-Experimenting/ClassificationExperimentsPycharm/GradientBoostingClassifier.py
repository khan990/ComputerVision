from numpy import genfromtxt
from sklearn.ensemble import GradientBoostingClassifier
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

    clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0,
                                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                 max_depth=5, init=None, random_state=None, max_features='sqrt', verbose=1,
                                 max_leaf_nodes=None, warm_start=False, presort='auto')


    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:, i] = clf.predict(Test_features.T);
    Total_Time = Total_Time + ((time.time() - start_time));
    # print(" %s seconds " % (time.time() - start_time))


print(" %s seconds " % Total_Time)
acc, sen, spe = ClassifierScore.CalculateScore(Test_Labels.T, predicted_labels)
ClassifierScore.PrintScore(acc, sen, spe)