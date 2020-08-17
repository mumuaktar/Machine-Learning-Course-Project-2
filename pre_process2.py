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

labels = np.zeros(25000);
labels[0:12500]=0;
labels[12500:25000]=1;
# np.save('movie_features_matrix.npy',features_matrix)
np.save('movie_labels.npy',labels)
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
                string_list.append(line)
                

cv = CountVectorizer(tokenizer=LemmaTokenizer(),
                                strip_accents = 'unicode', # works 
                                stop_words = 'english', # works
                                lowercase = True, # works
                                max_df = 0.95, # works
                                min_df = 10) # works
Data=cv.fit_transform(text_train)
word_count=np.sum(Data.toarray(),axis=1,keepdims = True)
binary_feature=Data.toarray()
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(Data) 
X=np.array(X_train_tfidf.toarray())
word_count=word_count/np.max(word_count)
X=np.concatenate((word_count,X),axis=1)                 
                   # dictionary_tmp = Counter(words)
                    #feature=d
#size_data=X.shape
#mi=np.zeros(size_data[1])
#si=np.zeros(size_data[1])
#for i in range(size_data[1]):
#    mi[i]=np.mean(X[:,i])
#    si[i]=np.std(X[:,i])
#X_norm=np.zeros(size_data,dtype=np.float32)
#for i in range(size_data[1]):
#    X_norm[:,i]=(X[:,i]-mi[i])/si[i]

binary_feature=Data.toarray()
binary_feature=binary_feature>0
np.save('Data_binary',binary_feature)    
np.save('X.npy',X)     
np.save('Label.npy',C) 
    