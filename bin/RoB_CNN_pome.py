import gc
import os
import csv
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix


from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from keras.callbacks import Callback

os.chdir('/home/qwang/qw_scripts/RoB')

# %% Text preparation
csv.field_size_limit(100000000)
dat0 = pd.read_csv("dataWithFullText_utf8.txt", sep='\t', engine="python", encoding="utf-8")    

# Delete invalid records
dat1 = dat0.copy()
dat1 = dat1[-dat1["ID"].isin([8, 608, 647, 703, 807, 903, 960, 1446, 1707, 1707, 1714, 1716, 1754, 2994, 2995, 2996, 2997, 3943, 4045, 4064, 4066, 4076, 4077, 4083, 3804, 4035])]
dat1.set_index(pd.Series(range(0, len(dat1))), inplace=True)

dat2 = dat1.copy()
# snow = SnowballStemmer("english")

for i, paper in enumerate(dat2["CleanFullText"]):
     # Remove non-ASCII characters
     temp = paper.encode("ascii", errors="ignore").decode()
     # Remove numbers
     temp = re.sub(r'\d+', '', temp)
     # Remove punctuations (warning: "-" should be replaced with '_')
     temp = re.sub(r'[^\w\s]','',temp)
     # Lower the cases
     temp = temp.lower()
     # Remove duplicate whitespaces
     temp = re.sub(r'\s+', ' ', temp)
     # Remove words that of 1 character length
     temp = re.sub(r'\b(\w{1})\b', '', temp)          # temp = re.sub(r'\b(\w{1,2})\b', '', temp)  
#     # Stemming
#     word_tokens = word_tokenize(temp) 
#     stem_list = [snow.stem(w) for w in word_tokens]     
#     # Convert list to string
#     filter_string = ' '.join(stem_list)   
     #print(i)
     dat2.loc[i,"CleanFullText"] = temp

dat = dat2.copy()


del(dat0)
del(dat1)
del(dat2)

# %% Train and split data
# dat = pd.read_csv("RoB_dat.csv")   
paper_train, paper_test, y_train, y_test = model_selection.train_test_split(dat['CleanFullText'], dat['RandomizationTreatmentControl'],
                                                                            test_size=0.2, random_state=66)
# dat['BlindedOutcomeAssessment']
# dat['SampleSizeCalculation']
gc.collect()


# %% Classification report and plot
plt.style.use('ggplot')


plt.style.use('seaborn-muted')
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


    

# plot_history(clf)


 
# %% Model 3: CNN-multichannel (YK)
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense
from keras.models import Model
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors



# ---------------------- text preparation ---------------------- 
tokenizer = Tokenizer(num_words=5000) # Only the most common num_words words will be kept based on word frequency.
tokenizer.fit_on_texts(paper_train)
X_train = tokenizer.texts_to_sequences(paper_train)
X_test = tokenizer.texts_to_sequences(paper_test)


vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
word_index = tokenizer.word_index


# ---------------------- Pad sequences ----------------------
doc_len = 1000
X_train = pad_sequences(X_train, padding='post', maxlen=doc_len)
X_test = pad_sequences(X_test, padding='post', maxlen=doc_len)

# ---------------------- Google's pre-trained Word2Vec model ----------------------
# word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
word_vectors = KeyedVectors.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary=True)


# ---------------------- Retrieve the embedding matrix ----------------------
embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))

num_words = 5000
for word, i in word_index.items():
    if i>=num_words:
        continue
    try:
        embedding_vector = word_vectors[word] # vector shape (300, 1)
        embedding_matrix[i] = embedding_vector[:embedding_dim]
    except KeyError:
        embedding_matrix[i]=np.random.normal(0, np.sqrt(0.25), embedding_dim)

del(word_vectors)


# ------------------------------------ Layers building ------------------------------------
input_doc = Input(shape=(doc_len,))
input_embed = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)(input_doc)
#input_doc = Input(shape=(doc_len, vocab_size)) # [doc_len, vocab_size] --> [doc_len, embedding_dim] = [n,d]
#input_embed = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=doc_len, trainable=True)(input_doc)  ### wrong

map_1 = Conv1D(filters=2, kernel_size=3, activation='relu')(input_embed)
map_1 = GlobalMaxPooling1D()(map_1)
map_1 = Dropout(rate=0.1)(map_1)

map_2 = Conv1D(filters=2, kernel_size=4, activation='relu')(input_embed)
map_2 = GlobalMaxPooling1D()(map_2)
map_2 = Dropout(rate=0.1)(map_2)

map_3 = Conv1D(filters=2, kernel_size=5, activation='relu')(input_embed)
map_3 = GlobalMaxPooling1D()(map_3)
map_3 = Dropout(rate=0.1)(map_3)

map = layers.concatenate([map_1, map_2, map_3], axis=1)
output = Dense(1, activation='sigmoid')(map)
model = Model(inputs=input_doc, outputs=output)



# ---------------------------------------------- Custom metric ----------------------------------------------
def sens(y_true, y_pred):
   y_pred_binary = K.switch(y_pred>0.5, K.ones_like(y_pred), K.zeros_like(y_pred))
   tp = K.sum(y_true * y_pred_binary)
   fn = K.sum(y_true * (1-y_pred_binary))
   sensitivity = K.switch(tp+fn>0, tp / (tp+fn), 0.0)
   return sensitivity     
        

def spec(y_true, y_pred):
   y_pred_binary = K.switch(y_pred>0.5, K.ones_like(y_pred), K.zeros_like(y_pred))
   tn = K.sum((1-y_true) * (1-y_pred_binary))
   fp = K.sum((1-y_true) * y_pred_binary)
   specificity = K.switch(tn+fp>0, tn / (tn+fp), 0.0)
   return specificity    

def prec(y_true, y_pred):
   y_pred_binary = K.switch(y_pred>0.5, K.ones_like(y_pred), K.zeros_like(y_pred))
   tp = K.sum(y_true * y_pred_binary)
   fp = K.sum((1-y_true) * y_pred_binary)
   precision = K.switch(tp+fp>0, tp / (tp+fp), 0.0)
   return precision    

def accu(y_true, y_pred):
   y_pred_binary = K.switch(y_pred>0.5, K.ones_like(y_pred), K.zeros_like(y_pred))
   tp = K.sum(y_true * y_pred_binary)
   fn = K.sum(y_true * (1-y_pred_binary))
   tn = K.sum((1-y_true) * (1-y_pred_binary))
   fp = K.sum((1-y_true) * y_pred_binary)
   accuracy = K.switch(tp+tn>0, (tp+tn) / (tp+tn+fp+fn), 0.0)
   return accuracy    

def f1(y_true, y_pred):
   y_pred_binary = K.switch(y_pred>0.5, K.ones_like(y_pred), K.zeros_like(y_pred))
   tp = K.sum(y_true * y_pred_binary)
   fn = K.sum(y_true * (1-y_pred_binary))
   fp = K.sum((1-y_true) * y_pred_binary)
   f1 = K.switch(2*tp+fp+fn>0, 2*tp / (2*tp+fp+fn), 0.0)
   return f1    
 
  
    
# ---------------------------------------------- Custom callback ----------------------------------------------    
class sens_spec(Callback):
    def __init__(self, x, y_true, num_classes, validation_data):
        super().__init__()
        #self.x = x
        #self.y_true 1= y_true
        self.validation_data = validation_data
        self.num_classes = num_classes

#    def on_epoch_end(self, epoch, logs=None):
#        # Compute the confusion matrix to get tp, fp, tn, fn
#        conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
#            
#        X_val, y_true = self.validation_data[0], self.validation_data[1]
#        y_pred_prob = np.asarray(model.predict(X_val))
#        y_pred = (y_pred_prob > 0.5).astype(int)       
#        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#        
#        sens = tp / (tp+fn)
#        spec = tn / (tn+fp)           
#        accu = (tp+tn) / (tp+tn+fp+fn)       
#        prec = tp / (tp+fp)       
#        f1 = 2*tp / (2*tp+fp+fn)
#
#        print("Sensitivity: {:.4f}".format(sens))
#        print("Specificity: {:.4f}".format(spec))
#        print("Accuracy: {:.4f}".format(accu))
#        print("Precision: {:.4f}".format(prec))
#        print("F1 score: {:.4f}".format(f1))
        
    def on_train_end(self, logs={}):
        # conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
            
        X_val, y_true = self.validation_data[0], self.validation_data[1]
        y_pred_prob = np.asarray(model.predict(X_val))
        y_pred = (y_pred_prob > 0.5).astype(int)       
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sens = tp / (tp+fn)
        spec = tn / (tn+fp)           
        accu = (tp+tn) / (tp+tn+fp+fn)       
        prec = tp / (tp+fp)       
        f1 = 2*tp / (2*tp+fp+fn)

        print("Sensitivity: {:.4f}".format(sens))
        print("Specificity: {:.4f}".format(spec))
        print("Accuracy: {:.4f}".format(accu))
        print("Precision: {:.4f}".format(prec))
        print("F1 score: {:.4f}".format(f1))




model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', sens, spec, accu, f1, prec])
# model.summary()
# clf = model.fit(X_train, y_train, epochs=50, batch_size=50, validation_data=(X_test, y_test), verbose=False)
num_classes = 2
sens_spec_callback = sens_spec(X_train, y_train, num_classes, validation_data=(X_test, y_test))
clf = model.fit(X_train, y_train, epochs=20, batch_size=500, validation_data=(X_test, y_test), callbacks=[sens_spec_callback])



loss, accuracy, sensitivity, specificity, custom_accuracy, f1, precision = model.evaluate(X_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
print("Training Sensitivity: {:.4f}".format(sensitivity))
print("Training Specificity: {:.4f}".format(specificity))
print("Training Custom Accuracy: {:.4f}".format(custom_accuracy))
print("Training f1 score: {:.4f}".format(f1))
print("Training Precision: {:.4f}".format(precision))
loss, accuracy, sensitivity, specificity, custom_accuracy, f1, precision = model.evaluate(X_test, y_test, verbose=True)
print("Test Accuracy: {:.4f}".format(accuracy))
print("Test Sensitivity: {:.4f}".format(sensitivity))
print("Test Specificity: {:.4f}".format(specificity))
print("Test Custom Accuracy: {:.4f}".format(custom_accuracy))
print("Test f1 score: {:.4f}".format(f1))
print("Test Precision: {:.4f}".format(precision))


plot_history(clf)














# %% Hyperparameters optimization
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, doc_len, drop_rate):
    
    input_doc = Input(shape=(doc_len,))
    input_embed = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)(input_doc)

    map_1 = Conv1D(filters = num_filters, kernel_size = kernel_size, activation='relu')(input_embed)
    map_1 = GlobalMaxPooling1D()(map_1)
    map_1 = Dropout(rate = drop_rate)(map_1)    
    map_2 = Conv1D(filters = num_filters, kernel_size = kernel_size+1, activation='relu')(input_embed)
    map_2 = GlobalMaxPooling1D()(map_2)
    map_2 = Dropout(rate = drop_rate)(map_2)  
    map_3 = Conv1D(filters = num_filters, kernel_size = kernel_size+2, activation='relu')(input_embed)
    map_3 = GlobalMaxPooling1D()(map_3)
    map_3 = Dropout(rate = drop_rate)(map_3)
    
    map = layers.concatenate([map_1, map_2, map_3], axis=1)
    output = Dense(1, activation='sigmoid')(map)
    model = Model(inputs=input_doc, outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', sens, spec, accu, f1, prec])
    return model


# Main settings
embedding_dim = 200
doc_len = 1000

epochs = 20
num_words = 5000
drop_rate = 0.1
output_file = 'output/rob_cnn_output.txt'

# text preparation 
tokenizer = Tokenizer(num_words=5000) # Only the most common num_words words will be kept based on word frequency.
tokenizer.fit_on_texts(paper_train)
X_train = tokenizer.texts_to_sequences(paper_train)
X_test = tokenizer.texts_to_sequences(paper_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
word_index = tokenizer.word_index


# Pad sequences 
X_train = pad_sequences(X_train, padding='post', maxlen=doc_len)
X_test = pad_sequences(X_test, padding='post', maxlen=doc_len)

# pre-trained word2vec
word_vectors = KeyedVectors.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary=True)

# Retrieve the embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if i>=num_words:
        continue
    try:
        embedding_vector = word_vectors[word] # vector shape (200, 1)
        embedding_matrix[i] = embedding_vector[:embedding_dim]
    except KeyError:
        embedding_matrix[i]=np.random.normal(0, np.sqrt(0.25), embedding_dim)
del(word_vectors)


# Parameter grid for grid search
param_grid = dict(num_filters=[2],
                  kernel_size=[7],
                  vocab_size=[vocab_size],
                  embedding_dim=[embedding_dim],
                  doc_len=[doc_len],
                  drop_rate=[drop_rate])

#num_classes = 2
#sens_spec_callback = sens_spec(X_train, y_train, num_classes, validation_data=(X_test, y_test))
#
#fit = dict(callbacks=sens_spec_callback)


model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=500, verbose=True)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, verbose=1, n_iter=5)

grid_result = grid.fit(X_train, y_train)

# Evaluate testing set
test_accuracy = grid.score(X_test, y_test)

# Save and evaluate results
s = ('Best Accuracy : ' 
         '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
output_string = s.format(grid_result.best_score_, test_accuracy, grid_result.best_params_)
print(output_string)


























