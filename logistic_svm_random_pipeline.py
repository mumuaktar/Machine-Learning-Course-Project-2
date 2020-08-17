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
                string_list.append(line)
                
                
C=np.load('movie_labels.npy')
stop_words = set(stopwords.words('english')) 
string = [w for w in string_list if not w in stop_words]

##train test splitting
text_train, text_test, y_train, y_test = train_test_split(
    string, C, test_size=0.10, random_state=45)

##functions: tokenization
cv = CountVectorizer(tokenizer=LemmaTokenizer(),
strip_accents = 'unicode', 
lowercase = True, 
max_df = 0.95, 
min_df = 25, ngram_range=(1,1)) 



start1 = time.time()
from sklearn.pipeline import Pipeline
pclf = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf_transformer', TfidfTransformer()),
    ('normalizer_transformer', Normalizer()),
    ('clf', LogisticRegression(penalty = 'l2')),
])
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.005,0.001, 0.01, 0.1, 1, 10]}   
grid = GridSearchCV(pclf, param_grid,cv=10)
grid.fit(X=text_train, y=y_train)

y_pred = grid.predict(text_test)
accuracy_score(y_test, y_pred)
end1= time.time()
print("accuracy for logistic regression: " + str(accuracy_score) + " which took: " + str((end1-start1)) + " s" + "")