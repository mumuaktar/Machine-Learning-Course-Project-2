# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:10:20 2019

@author: mumu
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
nltk.download('punkt')
nltk.download('wordnet')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


C=np.load('movie_labels.npy')


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
                

    #Convert @username to AT_USER
   
cv = CountVectorizer(tokenizer=LemmaTokenizer(),
                                #ngram_range=(2, 2),
                                binary=True,
                                strip_accents = 'unicode', # works 
                                stop_words = 'english', # works
                                lowercase = True, # works
                                max_df = 0.95, # works
                                min_df = 10) # works
Data=cv.fit_transform(string_list)
word_count=np.sum(Data.toarray(),axis=1,keepdims = True)
binary_feature=Data.toarray()




binary_feature=binary_feature>0
np.save('Data_binary_t',binary_feature) 
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(Data) 
X=np.array(X_train_tfidf.toarray())
word_count=word_count/np.max(word_count)
X=np.concatenate((word_count,X),axis=1)      
features=X;
labels=C;

np.save('features',features)

kf = StratifiedKFold(n_splits=10)
for train_index, test_index in kf.split(features,labels):
    
    X_train = [features[i] for i in train_index]
    X_test = [features[i] for i in test_index]
    
y_train, y_test = labels[train_index], labels[test_index]
#X_train, X_test, y_train, y_test  = train_test_split(
        #features, 
      #  C,
       # train_size=0.80, 
        #random_state=1234)
        #from sklearn.model_selection import GridSearchCV
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
#grid = GridSearchCV(LogisticRegression(), param_grid, cv=10)
#grid.fit(X_train, y_train)
#print("Best cross-validation score: {:.2f}".format(grid.best_score_))
#print("Best parameters: ", grid.best_params_)
#print("Best estimator: ", grid.best_estimator_)
def Bernoulli(): 
    logprior={}
    p={}
    likelihood={}
    N_doc = len(binary_feature)

    max_ = float("-inf")
    all_classes = set(labels)
    for c in all_classes:
        N_c = float(sum(labels == c))
        logprior[c] = np.log(N_c / N_doc)
        #p[c] = binary_feature[X_test==c].sum(0) + 1
        #likelihood[c]=np.log(p/p.sum()+2)
    #log_odds = np.log((p[all_classes==1]/likelihood[all_classes==1]) / (p[all_classes==0]/likelihood[all_classes==0]))
     p = binary_feature[X_test==1].sum(0) + 1
q = binary_feature[X_test==0].sum(0) + 1
likelihood_1=np.log(p/p.sum()+2)
#likelihood_0=np.log((q/q.sum()+2)   
    if log_odds > max_:
                max_ = log_odds
                pred_class = l
                return pred_class
            
     nb = Bernoulli()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
print(accuracy_score(y_test, y_pred))

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
#from sklearn.naive_bayes import BernoulliNB
#b=BernoulliNB()
#b=b.fit(X=X_train, y=y_train)
#b_pred=b.predict(X_test)
#print(accuracy_score(y_test, b_pred))