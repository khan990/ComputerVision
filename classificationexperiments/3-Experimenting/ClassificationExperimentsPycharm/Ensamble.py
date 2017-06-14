from numpy import genfromtxt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import ClassifierScore
import time
import numpy
from numpy import savetxt



Train_features = genfromtxt('Train_features', dtype='float', delimiter=',')
Train_Labels = genfromtxt('Train_Labels', dtype='int', delimiter=',')
Test_features = genfromtxt('Test_features', dtype='float', delimiter=',')
Test_Labels = genfromtxt('Test_Labels', dtype='int', delimiter=',')


predicted_labels = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_final = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_final_2 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
All_zeros = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');

predicted_labels_1 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_2 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_3 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_4 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_5 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_6 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_7 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_8 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');
predicted_labels_9 = numpy.zeros((len(Train_Labels.T), len(Train_Labels)), dtype='int');


predicted_labels_final = All_zeros;
predicted_labels_final_2 = All_zeros;

count_1 = 0;
count_2 = 0;
count_3 = 0;
count_4 = 0;
count_5 = 0;
count_6 = 0;
count_7 = 0;
count_8 = 0;
count_9 = 0;

Total_Time = 0;
Num_of_Ensambles = 0;

start_time = time.time();

#-------------------------
#AdaBoostClassifier

print("AdaBoostClassifier");
predicted_labels = All_zeros;
Num_of_Ensambles +=1;
for i in range(len(Train_Labels)):

    clf = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R',
                             random_state=None);

    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:, i] = clf.predict(Test_features.T);
    Total_Time = Total_Time + ((time.time() - start_time));
    # print(" %s seconds " % (time.time() - start_time))

predicted_labels_1 = numpy.int_(predicted_labels);
#predicted_labels_final += numpy.int_(predicted_labels);
print(" %s seconds since start " % (Total_Time));
#-----------------------------
#DecisionTreeClassifier
print("DecisionTreeClassifier");
predicted_labels = All_zeros;
Num_of_Ensambles +=1;
clf = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=None, min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0, max_features='auto', random_state=None, max_leaf_nodes=None,
                             class_weight='balanced', presort=False)
clfs = clf.fit(Train_features.T, Train_Labels.T)
start_time = time.time();
predicted_labels = clf.predict(Test_features.T)
Total_Time = Total_Time + ((time.time() - start_time));
predicted_labels_2 = numpy.int_(predicted_labels);
#predicted_labels_final += numpy.int_(predicted_labels);
print(" %s seconds since start " % (Total_Time));
#-----------------------------
#GaussianNB
print("GaussianNB");
predicted_labels = All_zeros;
Num_of_Ensambles +=1;
for i in range(len(Train_Labels)):

    clf = GaussianNB()
    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:, i] = clf.predict(Test_features.T);
    Total_Time = Total_Time + ((time.time() - start_time));

#predicted_labels_final += numpy.int_(predicted_labels);
predicted_labels_3 = numpy.int_(predicted_labels);
print(" %s seconds since start " % (Total_Time));
#-----------------------------
#GaussianProcess
'''
print("GaussianProcess");
predicted_labels = All_zeros;
Num_of_Ensambles +=1;
for i in range(len(Train_Labels)):

    clf = GaussianProcess(regr='constant', corr='squared_exponential', beta0=None, storage_mode='full',
                          verbose=True, theta0=0.1, thetaL=None, thetaU=None, optimizer='fmin_cobyla',
                          random_start=1, normalize=True, nugget=2.2204460492503131e-15, random_state=None)


    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:,i] = clf.predict(Test_features.T);
    Total_Time = Total_Time + ((time.time() - start_time));

#predicted_labels_final += numpy.int_(predicted_labels);
predicted_labels_4 = numpy.int_(predicted_labels);
print(" %s seconds since start " % (Total_Time));
'''
#-----------------------------
#GradientBoostingClassifier

print("GradientBoostingClassifier");
predicted_labels = All_zeros;
Num_of_Ensambles +=1;
for i in range(len(Train_Labels)):

    clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0,
                                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                 max_depth=5, init=None, random_state=None, max_features='sqrt', verbose=1,
                                 max_leaf_nodes=None, warm_start=False, presort='auto')


    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:, i] = clf.predict(Test_features.T);
    Total_Time = Total_Time + ((time.time() - start_time));

#predicted_labels_final += numpy.int_(predicted_labels);
predicted_labels_5 = numpy.int_(predicted_labels);
print(" %s seconds since start " % (Total_Time));

#-----------------------------
#LinearDiscriminantAnalysis
'''
print("LinearDiscriminantAnalysis");
predicted_labels = All_zeros;
Num_of_Ensambles +=1;
for i in range(len(Train_Labels)):
    clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                                     store_covariance=False, tol=0.0001)

    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:,i] = clf.predict(Test_features.T);

#predicted_labels_final += predicted_labels;
predicted_labels_6 = numpy.int_(predicted_labels);
print(" %s seconds since start " % (Total_Time));
'''
#-----------------------------
#QuadraticDiscriminantAnalysis
'''
print("QuadraticDiscriminantAnalysis");
predicted_labels = All_zeros;
Num_of_Ensambles +=1;
for i in range(len(Train_Labels)):

    clf = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariances=False, tol=0.0001)

    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:, i] = clf.predict(Test_features.T);

#predicted_labels_final += predicted_labels;
predicted_labels_7 = numpy.int_(predicted_labels);
print(" %s seconds since start " % (Total_Time));
'''
#-----------------------------
#RandomForestClassifier
print("RandomForestClassifier");
predicted_labels = All_zeros;
Num_of_Ensambles +=1;
clf = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=None, min_samples_split=10,
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                             max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=-1,
                             random_state=None, verbose=0, warm_start=False, class_weight=None)

clf2 = clf.fit(Train_features.T, Train_Labels.T)
start_time = time.time();
predicted_labels = clf.predict(Test_features.T)
Total_Time = Total_Time + ((time.time() - start_time));
#predicted_labels_final += numpy.int_(predicted_labels);
predicted_labels_8 = numpy.int_(predicted_labels);
print(" %s seconds since start " % (Total_Time));

#-----------------------------
#SVC
print("SupportVectorMachine");
predicted_labels = All_zeros;
Num_of_Ensambles +=1;

for i in range(len(Train_Labels)):

    clf = SVC(C=0.025, kernel='rbf', degree=1, gamma=2, coef0=20.0, shrinking=False, probability=True, tol=0.0001,
              cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,
              random_state=None)


    clf2 = clf.fit(Train_features.T, Train_Labels[i].T);

    start_time = time.time();
    predicted_labels[:,i] = clf.predict(Test_features.T);
    Total_Time = Total_Time + ((time.time() - start_time));

predicted_labels_9 = numpy.int_(predicted_labels);
#predicted_labels_final += numpy.int_(predicted_labels);
print(" %s seconds since start " % (Total_Time));

#-------------------------
#Final Computation
print(" %s seconds since start " % (Total_Time));
print(" %s seconds " % (Total_Time));
print("Number of Algorithms = %s " % Num_of_Ensambles);

predicted_labels_final = numpy.int_(predicted_labels_1) + numpy.int_(predicted_labels_2) + numpy.int_(predicted_labels_3) + numpy.int_(predicted_labels_4) + numpy.int_(predicted_labels_5) + numpy.int_(predicted_labels_6) + numpy.int_(predicted_labels_7) + numpy.int_(predicted_labels_8) + numpy.int_(predicted_labels_9)

for r in range(len(Train_Labels.T)):
    for c in range(len(Train_Labels)):
        if(predicted_labels_final[r][c] == 1):
            count_1 += 1;
        if(predicted_labels_final[r][c] == 2):
            count_2 += 1;
        if (predicted_labels_final[r][c] == 3):
            count_3 += 1;
        if (predicted_labels_final[r][c] == 4):
            count_4 += 1;
        if (predicted_labels_final[r][c] == 5):
            count_5 += 1;
        if (predicted_labels_final[r][c] == 6):
            count_6 += 1;
        if (predicted_labels_final[r][c] == 7):
            count_7 += 1;
        if (predicted_labels_final[r][c] == 8):
            count_8 += 1;
        if (predicted_labels_final[r][c] == 9):
            count_9 += 1;

        if(predicted_labels_final[r][c] >= 3): #(Num_of_Ensambles/2)):
            predicted_labels_final_2[r][c] = 1;
        else:
            predicted_labels_final_2[r][c] = 0;

#predicted_labels_final_2 = predicted_labels_final > (Num_of_Ensambles/2);

acc, sen, spe = ClassifierScore.CalculateScore(Test_Labels.T, predicted_labels_final_2);
ClassifierScore.PrintScore(acc, sen, spe);
savetxt('predicted_labels_final.csv', predicted_labels_final.T, delimiter=",")

print("count 1 = %d " % count_1);
print("count 2 = %d " % count_2);
print("count 3 = %d " % count_3);
print("count 4 = %d " % count_4);
print("count 5 = %d " % count_5);
print("count 6 = %d " % count_6);
print("count 7 = %d " % count_7);
print("count 8 = %d " % count_8);
print("count 9 = %d " % count_9);


