import pandas as pd
import numpy as np


"""
Use this toolbox either as an imported module, or in e.g. ipython with the magic %paste command 
to import w.e. is in you clipboard; that could be the jaccard function or one of the other tools.

Note to self:
Itemset finder with regex

^\s+([01]\s+){1}1\s+([01]\s+){4}1
"""



def jaccard(a,b):
    _11 = 0
    _not00 = 0
    for x,y in zip(a,b):
        if x and y:
            _11 += 1
            _not00 += 1
        elif x or y:
            _not00 += 1
    return float(_11)/float(_not00)

"""
Use of jaccard:
for i1 in range(5):
    for i2 in range(5):
        if i1 >= i2:
            continue
        print i1+1, i2+1, 1 - jaccard(_ns[i1], _ns[i2])
"""


def gini(vec):
    # It is 1 minus the sum over i of p_i^2, where p_i is the fraction of records belonging to class i.
    asseries = pd.Series(vec)
    valc = asseries.value_counts()
    normed = valc / len(vec)
    powered = normed * normed
    summed = powered.sum()
    return (1 - summed)

def classification_error(vec):
    return 1 - float(pd.Series(vec).value_counts()[0])/len(vec)

def purity_gain(parent, children, measure_method=gini):
    """
    Usage: purity_gain([1,0,1,0], [[0,0],[1,1]], gini)
    """
    # break early:
    children[0][1] # breaks if you're e.g. just passing a 1d list of chars
    #It is the impurity of the parent minus the sum over i of 
    #   (the number of records associated with the child node i divided by the total number of records in the parent node, 
    #   multiplied by the impurity measure of the child node i)
    pval = measure_method(parent)
    pl = float(len(parent))
    chvals = [measure_method(x) * len(x) / pl for x in children]
    
    return pval - sum(chvals)


def least_square(A, y):
    """
    Intercept is placed first
    """
    A = np.vstack([np.ones(len(A)), A]).T
    return np.linalg.lstsq(A, y)[0]


def standardise(vec, ddof=0):
    """
    subtract mean, then divide by standard deviation
    ddof is used as N - ddof in the divisor in std
    """
    vm = np.mean(vec)
    vs = np.std(vec, ddof=ddof)
    return [(x - vm)/vs for x in vec]

def pca_accountability(singular_values):
    """
    Answers the eternal question: How much variability does the nth principal component account for?
    Returns the fraction of variability accounting for each of the elements in singular_values, order preserved.
    """

    squared = [x*x for x in singular_values]
    svsum = float(sum(squared))
    ratio = [x / svsum for x in squared]
    return ratio


def classification_stats(TP, TN, FP, FN):
    count = TP + TN + FP + FN
    error_rate = float(FP + FN) / count
    TPR = TP / float(TP + FN)
    FPR = FP / float(TN + FP)
    FNR = FN / float(TP + FN)
    TNR = TN / float(TN + FP)
    sensitivity = TPR
    specificity = TN
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    F1 = 2 * TP / float(2 * TP + FP + FN)

    output = {
        'count': count,
        'error_rate': error_rate,
        'TPR': TPR,
        'FPR': FPR,
        'FNR': FNR,
        'TNR': TNR,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'F1': F1
    }

    return output


# Wrapper methods

def mean(vals):
    return pd.Series(vals).mean()

def median(vals):
    return pd.Series(vals).median()

def mode(vals):
    return pd.Series(vals).mode()

def valuerange(vals):
    ser = pd.Series(vals)
    return ser.max() - ser.min()


