import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.decomposition import NMF 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.cross_validation import train_test_split 


def tfidf_matrix(series):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(series)
	#create tfidf matrix from series  
    #create reverse lookup of tokens 
    return tfidf_mat

def random_forest(X_train, y_train): 
    rf = RandomForestRegressor()
    rf.fit_transform(X_train, y_train) 
    return rf 

def gradient_boosting(series): 
    pass 

if __name__=='__main__':	
	df = pd.read_pickle('df_tokenized.pkl')
    #create tfidf matrix for total_text
    #create tfidf matrix for titles 
    text_mat = tfidf_matrix(df['total_text_word_tokens'])
    title_mat = tfidf_matrix(df['title_word_tokens'])
    #combine matrices 
    total_mat = np.hstack((text_mat, title_mat))
    #create dummies for category_code 
    resp = df['age']
    X_train, X_test, y_train, y_test = train_test_split(total_mat, resp, test_size = 0.3)

    #random forest
    rf = random_forest(X_train, y_train)
    rf.score(X_test, rf.predict(X_test))
	
