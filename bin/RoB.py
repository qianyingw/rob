import gc
import os
os.chdir('S:/TRIALDEV/CAMARADES/Qianying/RoB_All')
# import RoB_data_helpers

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import make_scorer, recall_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
#from sklearn import svm
#from sklearn.svm import SVC
#from sklearn.svm import LinearSVC


# split the data into training and test sets
dat = pd.read_csv("RoB_dat.csv")   
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(dat['CleanFullText'], 
                                                                      dat['RandomizationTreatmentControl'],
                                                                      test_size = 0.2, random_state = 66)
#X_train, X_valid, y_train, y_valid = model_selection.train_test_split(dat['CleanFullText'], dat['BlindedOutcomeAssessment'])
#X_train, X_valid, y_train, y_valid = model_selection.train_test_split(dat['CleanFullText'], dat['SampleSizeCalculation'])

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(dat['CleanFullText'], 
                                                                      dat['SampleSizeCalculation'],
                                                                      test_size = 0.2, random_state = 66)


# %% Grid search (Multiple output)
if __name__ == "__main__":
    # find the best parameters for both the feature extraction and the classifier    
    # define random_state in SGDClssifier
    seeds = [66]
    sensitivity_score = make_scorer(recall_score, greater_is_better=True)   
    # store prediction arrays of different estimators on the test sets
    pred = pd.DataFrame(np.zeros((len(y_valid), len(seeds))), columns=list(seeds))
    # classification report matrix
    report = pd.DataFrame(np.zeros((len(seeds),6)), index = seeds,
                          columns=['Sensitivity','Specificity','Accuracy','Precision','f1', 'Validation_score'])   
          
    for seed in seeds:
        pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words='english', token_pattern=r'\w{1,}', min_df=2, ngram_range=(1,2))),
            ('tfidf', TfidfTransformer(use_idf=True, norm=None, sublinear_tf=True)),
            ('clf_sgd', SGDClassifier(random_state=seed, max_iter=50, alpha=0.0001, penalty='l2')),
        ])            
        parameters = { 'vect__max_features': (1100, ), }
    
        clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1, scoring=sensitivity_score)
        clf.fit(X_train, y_train) 
        y_true, y_pred = y_valid, clf.predict(X_valid)
        
        pred.loc[:, seed] = y_pred

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()      
        report.loc[seed, 'Sensitivity'] = tp / (tp+fn)
        report.loc[seed, 'Specificity'] = tn / (tn+fp)
        report.loc[seed, 'Accuracy'] = (tp+tn) / (tp+fp+fn+tn)
        report.loc[seed, 'Precision'] = tp / (tp+fp)
        report.loc[seed, 'f1'] = 2*tp / (2*tp + fp + fn)
        report.loc[seed, 'Validation_score'] = clf.best_score_

report.to_csv('report.csv')

clf.best_estimator_.get_params()
# Save best estimators
joblib.dump(clf.best_estimator_, 'SGD_seed10.pkl')


# %% Feature Extraction: Bag of Words
count_vect = CountVectorizer(stop_words='english', token_pattern=r'\w{1,}', min_df=2, ngram_range=(1,2), max_features=1200)
X_train_count = count_vect.fit_transform(X_train)
X_valid_count = count_vect.fit_transform(X_valid)

tfidf = TfidfTransformer(use_idf=True, norm=None, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train_count)
X_valid_tfidf = tfidf.fit_transform(X_valid_count)


tfidf_vect = TfidfVectorizer(stop_words='english', token_pattern=r'\w{1,}', min_df=2, ngram_range=(1,2),  max_features=1200,
                             sublinear_tf=True, use_idf=True, norm=None)
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_valid_tfidf = tfidf_vect.fit_transform(X_valid)


tfidf_vect = TfidfVectorizer(stop_words='english', token_pattern=r'\w{1,}', min_df=2, ngram_range=(1,2),  max_features=1000,
                             sublinear_tf=True, use_idf=True, norm=None)
X_train_tfidf = tfidf_vect.fit(X_train)
X_valid_tfidf = tfidf_vect.fit(X_valid)
   
def report(y, y_predict):
    tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    accuracy = (tp+tn) / (tp+fp+fn+tn)
    print("\n Sensitivity: %0.3f" % sensitivity)
    print(" Specificity: %0.3f" % specificity)
    print(" Accuracy: %0.3f" % accuracy)  


# %% SGD (non)
sgd = SGDClassifier(random_state=42, max_iter=50, alpha=0.0001, penalty='l2')
sgd.fit(X_train_tfidf, y_train) 
y_valid_predict = sgd.predict(X_valid_tfidf)
report(y_valid, y_valid_predict)



# %% 1. Pipeline of BoW and SGD 
#pipeline = Pipeline([
#    ('vect', CountVectorizer()),
#    ('tfidf', TfidfTransformer()),
#    ('clf_sgd', SGDClassifier()),
#])
#       
#parameters = {
#    'vect__min_df': (2, 4, 8),
#    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#    'vect__token_pattern': (r'\w{1,}',),
#    'vect__max_features': (100, 500, 1000),
#    'vect__stop_words': ('english',),
#    
#    'tfidf__use_idf': (True, False),
#    'tfidf__norm': ('l1', 'l2', None),
#    'tfidf__sublinear_tf': (True, False),
#    
#    'clf_sgd__alpha': (0.01, 0.001, 0.0001),
#    'clf_sgd__penalty': ('l2', 'elasticnet'),
#    'clf_sgd__max_iter': (50,),
#    'clf_sgd__random_state': (42,),
#}
                           

pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english', token_pattern=r'\w{1,}', min_df=2, ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True, norm=None, sublinear_tf=True)),
    ('clf_sgd', SGDClassifier(random_state=30, max_iter=50, alpha=0.0001, penalty='l2')),
])

        
parameters = {
    'vect__max_features': (1100, ),
}

gc.collect()



# %% Grid search
def clf_report(clf, X_valid, y_valid):
    print("\n Best parameters set found on development set: %r \n" % clf.best_params_)
    print(" Grid scores on development set:")
    
    for mean, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params']):
        print(" %0.3f for %r " % (mean, params.values()))
    print("\n Detailed classification report: ")
    
    y_true, y_pred = y_valid, clf.predict(X_valid)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    precision = tp / (tp+fp)
    f1 = 2*tp / (2*tp + fp + fn)    
    
    print(" Sensitivity  Specificity  Accuracy  Precision  F1_Score\n %0.3f        %0.3f        %0.3f     %0.3f      %0.3f \n" 
          % (sensitivity, specificity, accuracy, precision, f1))


sensitivity = make_scorer(recall_score, greater_is_better=True)

clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-2, verbose=1, scoring=sensitivity)
clf.fit(X_train, y_train)
clf_report(clf, X_valid, y_valid)













