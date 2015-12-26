import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.decomposition import NMF 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.cross_validation import train_test_split 


def tfidf_matrix(series):
	#create tfidf matrix from series  
    #create reverse lookup of tokens 
    

def random_forest(series): 
    pass 

def gradient_boosting(series): 
    pass 

if __name__=='__main__':	
	df = pd.read_pickle('df_age_predict_tokenized.pkl')
    #create tfidf matrix for total_text
    #create tfidf matrix for titles 
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['total_text'])
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, resp, test_size = 0.3)
    rf = RandomForestRegressor()
    rf.fit_transform(X_train, y_train) 
    

	
