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


def preprocess(text):
    #Convert to lower case
    text = text.lower()
    #Convert www.* or https?://* to URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)
    #Convert @username to AT_USER
    text = re.sub('@[^\s]+','AT_USER',text)
# split into tokens by white space
    tokens = text.split()# remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
# filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
# filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def make_Dictionary(root_dir):
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    all_words = []       
    for emails_dir in emails_dirs:
        emails = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for mail in emails:
            with open(mail,encoding="utf8") as m:
                for line in m:
                    words = preprocess(line)
                    all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    
    np.save('all_words.npy',all_words) 
    
    np.save('dict_movie.npy',dictionary) 
    
    return dictionary
    

root_dir = 'C:/Users/User/train'
dictionary = make_Dictionary(root_dir)

labels = np.zeros(25000);
labels[0:12500]=0;
labels[12500:25000]=1;
# np.save('movie_features_matrix.npy',features_matrix)
np.save('movie_labels.npy',labels)
