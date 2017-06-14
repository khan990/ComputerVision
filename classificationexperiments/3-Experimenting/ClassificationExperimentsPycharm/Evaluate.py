import math



def Evaluate(Actual, Predicted):

    idx = (Actual() == 1);

    p = len(Actual(idx));
    n = len(Actual(~idx));
    N = p + n;

    tp = sum(Actual(idx) == Predicted(idx));
    tn = sum(Actual(~idx) == Predicted(~idx));
    fp = n - tn;
    fn = p - tp;

    tp_rate = tp / p;
    tn_rate = tn / n;

    accuracy = (tp + tn) / N;
    sensitivity = tp_rate;
    specificity = tn_rate;
    precision = tp / (tp + fp);
    recall = sensitivity;
    f_measure = 2 * ((precision * recall) / (precision + recall));
    gmean = math.sqrt(tp_rate * tn_rate);

    return [accuracy, sensitivity, specificity]