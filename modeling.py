import numpy as np
import scipy as sc
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.decomposition import NMF 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.cross_validation import train_test_split 


def tfidf_matrix(series):
    vectorizer = TfidfVectorizer()
    tfidf_mat = vectorizer.fit_transform(series)
	#create tfidf matrix from series  
    #create reverse lookup of tokens 
    return tfidf_mat

def random_forest(X_train, y_train): 
    rf = RandomForestRegressor(n_jobs = -1)
    rf.fit_transform(X_train, y_train) 
    return rf 

def gradient_boosting(X_train, y_train): 
    gb = GradientBoostingRegressor(n_jobs = -1)
    gb.fit_transform(X_train, y_train) 
    return gb 
    

if __name__=='__main__':	
    df = pd.read_pickle('df_age_predict.pkl')
    #create tfidf matrix for total_text
    #create tfidf matrix for titles 
    text_mat = tfidf_matrix(df['total_text'])
    title_mat = tfidf_matrix(df['title'])
    #create dummies for category_code
    dummies = pd.get_dummies(df['category_code'])
    dummies.drop('m4m', axis=1, inplace=True)    
    #combine matrices 
    total_mat = sc.sparse.hstack((text_mat, title_mat, np.array(dummies)))
    resp = df['age']
    X_train, X_test, y_train, y_test = train_test_split(total_mat, resp, test_size = 0.3)

    #random forest
    #rf = random_forest(X_train, y_train)
    #rf.score(X_test, rf.predict(X_test))
	
