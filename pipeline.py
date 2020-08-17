# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:52:55 2019

@author: umroot
"""

import re
import os
import string
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.preprocessing import Normalizer
import time


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
    
    
root_dir = 'C:\mumu\comp551\PROJECT2\data'
emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
all_words = [] 
string_list=[]      
for emails_dir in emails_dirs:
    emails = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
    for mail in emails:
        with open(mail,encoding="utf8") as m:
            for line in m:
                line = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',line)
                line = re.sub('@[^\s]+','AT_USER',line)
                string_list.append(line)
                
                
C=np.load('movie_labels.npy')
stop_words = set(stopwords.words('english')) 
string_data = [w for w in string_list if not w in stop_words]

##train test splitting
text_train, text_test, y_train, y_test = train_test_split(
    string_data, C, test_size=0.10, random_state=45)

##functions: tokenization
cv = CountVectorizer(tokenizer=LemmaTokenizer(),
strip_accents = 'unicode', 
lowercase = True, 
max_df = 0.95, 
min_df = 5, ngram_range=(1,2)) 


start1 = time.time()
from sklearn.pipeline import Pipeline
logistic = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf_transformer', TfidfTransformer()),
    ('normalizer_transformer', Normalizer()),
    ('clf', LogisticRegression(penalty = 'l2')),
])
    
from sklearn.model_selection import GridSearchCV
param = {'clf__C': [0.001, 0.01, 0.1, 1, 2,4,6,8]}   
grid = GridSearchCV(logistic, param_grid=param,cv=10)
grid.fit(X=text_train, y=y_train)

y_pred = grid.predict(text_test)
#accuracy_score(y_test, y_pred)
end1= time.time()
print("accuracy for logistic regression: " + str(accuracy_score(y_test, y_pred))+ " which took: " + str((end1-start1)) + " s" + "")
print(grid.best_params_)

start2 = time.time()
from sklearn.pipeline import Pipeline
svm_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf_transformer', TfidfTransformer()),
    ('normalizer_transformer', Normalizer()),
    ('clf', LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=3.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)),
])

svm_pipe.fit(X=text_train, y=y_train)
y_pred = svm_pipe.predict(text_test)
#accuracy_score(y_test, y_pred)
end2= time.time()
print("accuracy for SVM: " + str(accuracy_score(y_test, y_pred)) + " which took: " + str((end2-start2)) + " s" + "")
