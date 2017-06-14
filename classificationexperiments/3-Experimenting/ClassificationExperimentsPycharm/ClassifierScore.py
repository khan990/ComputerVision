import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import collections



def CalculateScore(Actual_Labels, Predicted_Labels):
    Actual_Labels_temp = np.ravel(Actual_Labels);
    Predicted_Labels_temp = np.ravel(Predicted_Labels);

    #[fpr, sensitivity, thresholds = roc_curve(Actual_Labels_temp, Predicted_Labels_temp);
    #specificity = 1 + fpr;

    Actual_arrayed = np.array(Actual_Labels_temp);
    #check = collections.Counter(Actual_arrayed);
    p = collections.Counter(Actual_arrayed)[1];
    n = collections.Counter(Actual_arrayed)[0];
    N = p + n;

    #Matches
    all_matches = find_equal(Actual_Labels_temp, Predicted_Labels_temp);
    tp = collections.Counter(all_matches)[1];
    tn = collections.Counter(all_matches)[0];
    fp = n-tn;
    fn = p-tp;

    tp_rate = tp/p;
    tn_rate = tn/n;

    accuracy = (tp + tn) / N;
    sensitivity = tp_rate;
    specificity = tn_rate;
#    precision = tp / (tp + fp);
#    recall = sensitivity;
#    f_measure = 2 * ((precision * recall) / (precision + recall));

    #accuracy = accuracy_score(Actual_Labels, Predicted_Labels);
    return accuracy, sensitivity, specificity;


def PrintScore(accuracy, sensitivity, specificity):
    average = (sensitivity + specificity)/2;
    print("Accuracy: %f" % accuracy);
    print("Average: %f" % average);
    print("Sensitivity: %f" % sensitivity);
    print("Specificity: %f" % specificity);

def find_equal(a, b):
    result = []
    for i, x in enumerate(a):
        if x == b[i]:
            result.append(b[i])
    return result