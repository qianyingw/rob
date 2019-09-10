import gc

import os
import numpy as np
import pandas as pd
import csv
import re

#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
#from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from time import time
from sklearn.pipeline import Pipeline
from pprint import pprint

os.getcwd()
os.chdir('S:/TRIALDEV/CAMARADES/Qianying/SCR_NP')

#==============================================================================
# Text preparation
#==============================================================================
# dat0 = pd.read_csv("CAMARADESROBData1.txt", sep='\t', engine="python", encoding="latin-1")    
csv.field_size_limit(100000000)

Original_dataset_All_references_Sep2012.xlsx






dat0 = pd.read_csv("dataWithFullText_utf8.txt", sep='\t', engine="python", encoding="utf-8")    

# Delete invalid records
dat1 = dat0.copy()
dat1 = dat1[-dat1["ID"].isin([8, 608, 647, 703, 807, 903, 960, 1446, 1707, 1707, 1714, 1716, 1754, 2994, 2995, 2996, 2997, 3943, 4045, 4064, 4066, 4076, 4077, 4083, 3804, 4035])]
dat1.set_index(pd.Series(range(0, len(dat1))), inplace=True)


dat2 = dat1.copy()
snow = SnowballStemmer("english")


for i, paper in enumerate(dat2["CleanFullText"]):
     # Remove non-ASCII characters
     temp = paper.encode("ascii", errors="ignore").decode()
     # Remove numbers
     temp = re.sub(r'\d+', '', temp)
     # Remove punctuations (warning: "-" should be replaced by whitespace)
     temp = re.sub(r'[^\w\s]','',temp)
     # Lower the cases
     temp = temp.lower()
     # Remove duplicate whitespaces
     temp = re.sub(r'\s+', ' ', temp)
     # Remove words that of 1 or 2 charactes length
     temp = re.sub(r'\b(\w{1,2})\b', '', temp)            
     # Stemming
     word_tokens = word_tokenize(temp) 
     stem_list = [snow.stem(w) for w in word_tokens]     
     # Convert list to string
     filter_string = ' '.join(stem_list)   
     #print(i)
     dat2.loc[i,"CleanFullText"] = filter_string


dat = dat2.copy()   

gc.collect()

#==============================================================================
# split the data into training and validation datasets
#==============================================================================
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(dat['CleanFullText'], 
                                                                      dat['RandomizationTreatmentControl'],
                                                                      test_size = 0.2, random_state = 66)
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(dat['CleanFullText'], dat['BlindedOutcomeAssessment'])
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(dat['CleanFullText'], dat['SampleSizeCalculation'])



#==============================================================================
# Feature Extraction: Bag of Words
#==============================================================================
count_vect = CountVectorizer(stop_words='english', token_pattern=r'\w{1,}', min_df=8, ngram_range=(1,3), max_features=500)
X_train_count = count_vect.fit_transform(X_train)
X_valid_count = count_vect.fit_transform(X_valid)

tfidf = TfidfTransformer(use_idf=True, norm='l2', sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train_count)
X_valid_tfidf = tfidf.fit_transform(X_valid_count)


#tfidf_vect = TfidfVectorizer(sublinear_tf=True, use_idf=True,
#                             min_df=8, ngram_range=(1,3), token_pattern=r'\w{1,}', max_features=500,
#                             norm='l2', stop_words='english')
#X_train_tfidf = tfidf_vect.fit_transform(X_train)
#X_valid_tfidf = tfidf_vect.fit_transform(X_valid)


def report(y, y_predict):
    tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()
    sensitivity = np.round(tp / (tp+fn) * 100, 2)
    specificity = np.round(tn / (tn+fp) * 100, 2)
    accuracy = np.round((tp+tn)/(tp+fp+fn+tn) * 100, 2)
    print("sensitivity", sensitivity)
    print("specificity", specificity)
    print("accuracy", accuracy)
    


#==============================================================================
# SGD (non)
#============================================================================== 
sgd = SGDClassifier(alpha=.0001, max_iter=50, penalty="elasticnet")

sgd.fit(X_train_tfidf, y_train) 
y_valid_predict = sgd.predict(X_valid_tfidf)
report(y_valid, y_valid_predict)


#==============================================================================
# 1. Pipeline of BoW and SGD 
#==============================================================================
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf_sgd', SGDClassifier()),
])

        
parameters = {
    'vect__min_df': (2, 4, 8),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'vect__token_pattern': (r'\w{1,}',),
    'vect__max_features': (100, 500, 1000),
    'vect__stop_words': ('english',),
    
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2', None),
    'tfidf__sublinear_tf': (True, False),
    
    'clf_sgd__alpha': (0.01, 0.001, 0.0001),
    'clf_sgd__penalty': ('l2', 'elasticnet'),
    'clf_sgd__max_iter': (50,),
    'clf_sgd__random_state': (42,),
}
                           

pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english', token_pattern=r'\w{1,}', min_df=2, ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True, norm=None, sublinear_tf=True)),
    ('clf_sgd', SGDClassifier(random_state=42, max_iter=50, alpha=0.0001, penalty='l2')),
])

        
parameters = {
    'vect__max_features': (950, 1000, 1050, 1100, 1150,1200),
}

gc.collect()


#==============================================================================
# 2. Pipeline of BoW and SVM
#==============================================================================   
pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english', token_pattern=r'\w{1,}', 
                             min_df=2, ngram_range=(1,2), max_features=1100)),
    ('tfidf', TfidfTransformer(use_idf=True, norm=None, sublinear_tf=True)),
    ('clf', SVC(random_state=42, max_iter=-1)),
])

parameters = [{'clf__kernel': ['rbf'],                     
               'clf__C': [1, 10, 100, 1000],
               'clf__gamma': [1e-3, 1e-4]},
                    
#              {'clf__kernel': ['linear'], 
#               'clf__C': [1, 10, 100, 1000]},
              
              {'clf__kernel': ['poly'], 
               'clf__C': [1, 10, 100, 1000],
               'clf__gamma': [1e-3, 1e-4],
               'clf__degree': [2, 3]}
             ]

parameters = [{'clf__kernel': ['poly'],                     
               'clf__C': list(np.logspace(-2, 3, 6)),
               'clf__gamma': list(np.logspace(-2, 3, 6)),
               'clf__degree': [2, 3]} #,
                    
#              {'clf__kernel': ['linear'], 
#               'clf__C': [1, 10, 100, 1000]},
              
#              {'clf__kernel': ['poly'], 
#               'clf__C': [1, 100],
#               'clf__gamma': [1e-3],
#               'clf__degree': [2, 3]}
             ]

#==============================================================================
# Grid search
#==============================================================================
# scores = ['precision', 'recall']
scores = ['recall']
for score in scores:
    print("# Tuning hyper-parameters for %s \n" % score)
    # clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-2, verbose=1, scoring='%s_macro' % score)
    clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-2, verbose=1, scoring='recall_macro')
    clf.fit(X_train, y_train)
 
    print("Grid scores on development set:")
    print()

    for mean, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params']):
        print("%0.3f for %r \n" % (mean, params.values()))

    print("Best parameters set found on development set: \n")
    print(clf.best_params_)
    print(" \n Detailed classification report: \n")

    y_true, y_pred = y_valid, clf.predict(X_valid)
    # print(classification_report(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    precision = tp / (tp+fp)
    f1 = 2*tp / (2*tp + fp + fn)
    print("Sensitivity: %0.3f" % sensitivity)
    print("Specificity: %0.3f" % specificity)
    print("Accuracy: %0.3f" % accuracy)  
    print("Precision: %0.3f" % precision)
    print("F1 score: %0.3f \n" % f1)


















#==============================================================================
# LDA
#==============================================================================
from genism.test.utils import common_texts, datapath
from genism.corpora.dictionary import Dictionary
from genism import LdaModel
# create a corpus
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
# train the model
lda = LdaModel(common_corpus, num_topics=10)
# save model to disk
temp_file = datapath("model")
# load a pretrained model from disk
lda = LdaModel.load(temp_file)

other_texts = [
    ...       
]
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
unseen_doc = other_corpus[0]
vector = lda[unseen_doc] # 
lda.update(other_corpus)


lda = LdaModel(common_corpus, num_topics=50, alpha='auto', eval_every=5) 










                           
if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected block
    # find the best parameters for both the feature extraction and the classifier
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-2, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
y_valid_predict = grid_search.best_estimator_.predict(X_valid)
report(y_valid, y_valid_predict)













#==============================================================================
# NB
#==============================================================================
NB = MultinomialNB(alpha=0.0001, fit_prior=True, class_prior=None)
NB.fit(X_train_tfidf, y_train)
y_valid_predict = NB.predict(X_valid_tfidf)
report(y_valid, y_valid_predict)




def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):    
    classifier.fit(feature_vector_train, label) # fit the training data on the classifier
    predictions = classifier.predict(feature_vector_valid) # predict the labels on validation set
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    # classification report
    tn, fp, fn, tp = confusion_matrix(y_valid, predictions).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    accuracy = metrics.accuracy_score(predictions, y_valid)    
    return sensitivity, specificity, accuracy



#==============================================================================
# Linear Classifier 
#==============================================================================  
accuracy321 = train_model(linear_model.LogisticRegression(), X_train_count, y_train, X_valid_count)
accuracy322 = train_model(linear_model.LogisticRegression(), X_train_tfidf, y_train, X_valid_tfidf)
print("LR, Count Vectors: ", accuracy321)
print("LR, WordLevel TF-IDF: ", accuracy322)







#==============================================================================
# load the pre-trained word-embedding vectors
#==============================================================================
embeddings_index = {}
for i, line in enumerate(open('S:/TRIALDEV/CAMARADES/Qianying/RoB_DL_IICARus/wiki-news-300d-1M.vec', encoding='utf8')):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
# create a tokenizer
token = text.Tokenizer()
# Class for vectorizing texts, or/and turning texts into sequences 
  # (=list of word indexes, where the word of rank i in the dataset (starting at 1) has index i).
token.fit_on_texts(train['FullText'])
word_index = token.word_index # dictionary mapping words (str) to their rank/index. 
                              # Only set after fit_on_texts was called.

# convert text to sequence of tokens and pad them to ensure equal length vectors 
X_train_seq = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=70)
X_valid_seq = sequence.pad_sequences(token.texts_to_sequences(X_valid), maxlen=70)

# create token-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



#==============================================================================
# Convolutional Neural Network
#==============================================================================
def create_cnn():
    input_layer = layers.Input((70, ))
    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)
    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')    
    return model

classifier = create_cnn()
sensitivity, specificity, accuracy = train_model(classifier, X_train_seq, y_train, X_valid_seq, is_neural_net=True)
print("CNN, Word Embeddings",  accuracy)






