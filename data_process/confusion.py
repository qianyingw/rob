# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:47:53 2019

@author: s1515896
"""

from sklearn.metrics import confusion_matrix
def compute_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = 100 * 2*tp / (2*tp+fp+fn)
    accuracy = 100 * (tp+tn) / (tp+tn+fp+fn)
    recall = 100 * tp / (tp+fn)
    specificity = 100 * tn / (tn+fp)
    precision = 100 * tp / (tp+fp)
    print("f1: {0:.2f}% | accuracy: {1:.2f}% | sensitivity: {2:.2f}% | specificity: {3:.2f}% | precision: {4:.2f}%".format(
            f1, accuracy, recall, specificity, precision))
    #return f1, accuracy, recall, specificity, precision