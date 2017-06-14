from numpy import genfromtxt
from numpy import savetxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
import Evaluate
import time
import ClassifierScore

from sklearn.metrics import confusion_matrix


Train_features = genfromtxt('Train_features', dtype='float', delimiter=',')
Train_Labels = genfromtxt('Train_Labels', dtype='int', delimiter=',')
Test_features = genfromtxt('Test_features', dtype='float', delimiter=',')
Test_Labels = genfromtxt('Test_Labels', dtype='int', delimiter=',')



clf = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=None, min_samples_split=10,
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                             max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=-1,
                             random_state=None, verbose=0, warm_start=False, class_weight=None)

clf2 = clf.fit(Train_features.T, Train_Labels.T)

start_time = time.time();
predicted_labels = clf.predict(Test_features.T)
print(" %s seconds " % (time.time() - start_time))


acc, sen, spe = ClassifierScore.CalculateScore(Test_Labels.T, predicted_labels)
ClassifierScore.PrintScore(acc, sen, spe)