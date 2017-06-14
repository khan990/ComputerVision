from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import ClassifierScore
import time
from numpy import savetxt

Train_features = genfromtxt('Train_features', dtype='float', delimiter=',')
Train_Labels = genfromtxt('Train_Labels', dtype='int', delimiter=',')
Test_features = genfromtxt('Test_features', dtype='float', delimiter=',')
Test_Labels = genfromtxt('Test_Labels', dtype='int', delimiter=',')


clf = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0, max_features='auto', random_state=None, max_leaf_nodes=None,
                             class_weight='balanced', presort=False)

clfs = clf.fit(Train_features.T, Train_Labels.T)


start_time = time.time();
predicted_labels = clf.predict(Test_features.T)
print(" %s seconds " % (time.time() - start_time))


acc, sen, spe = ClassifierScore.CalculateScore(Test_Labels.T, predicted_labels)
ClassifierScore.PrintScore(acc, sen, spe)

#savetxt('tested_it.csv', predicted_labels.T, delimiter=",")