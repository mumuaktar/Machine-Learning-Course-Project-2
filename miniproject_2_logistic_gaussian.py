# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:08:28 2019

@author: umroot
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 01:33:45 2019

@author: User
"""

import os
import re
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

#nltk.download('punkt')
#nltk.download('wordnet')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


C=np.load('movie_labels.npy')
cleanr = re.compile('<.*?>')

root_dir = 'C:\\mumu\\comp551\\PROJECT2\\data'
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
                line = re.sub(cleanr, '', line)
                string_list.append(line)
                cv = CountVectorizer(tokenizer=LemmaTokenizer(),
                                strip_accents = 'unicode', # works 
                                #stop_words = 'english', # works
                                lowercase = True, # works
                                max_df = 0.98, # works
                                min_df = 10, ngram_range=(1,1)) # works

stop_words = set(stopwords.words('english')) 
string_list = [w for w in string_list if not w in stop_words]
Data=cv.fit_transform(string_list)
 

word_count=np.sum(Data.toarray(),axis=1,keepdims = True)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(Data) 
X=np.array(X_train_tfidf.toarray())
word_count=word_count/np.max(word_count)
X=np.concatenate((word_count,X),axis=1)      
labels=C;
X_train, X_test, y_train, y_test = train_test_split(
    string_list, labels, test_size=0.10, random_state=42)

normalizer_tranformer = Normalizer().fit(X=X_train)
X_train_normalized = normalizer_tranformer.transform(X_train)
X_test_normalized = normalizer_tranformer.transform(X_test)

from sklearn.naive_bayes import GaussianNB
G = GaussianNB()
G.fit(X=X_train, y=y_train)
G_pred=G.predict(X_test)
print(accuracy_score(y_test, G_pred))

###Logistic
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.005,0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(penalty = 'l2'), param_grid)
grid.fit(X_train, y_train)
#log_model = LogisticRegression()
#log_model = log_model.fit(X=X_train, y=y_train)
y_pred = grid.predict(X_test)
print(accuracy_score(y_test, y_pred))