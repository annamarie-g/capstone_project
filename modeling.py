import numpy as np
import scipy as sc
import pandas as pd 
import cPickle
import text_preprocessing as tp
from sklearn.cluster import KMeans 
from sklearn.decomposition import NMF 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.cross_validation import train_test_split 


def tfidf_matrix(series):
    vectorizer = TfidfVectorizer(tokenizer = tp.custom_tokenizer, stop_words=tp.custom_stop_words(), lowercase=True)
    tfidf_mat = vectorizer.fit_transform(series)
	#create tfidf matrix from series  
    #create reverse lookup of tokens 
    features = vectorizer.get_feature_names()
    return tfidf_mat, features

def build_feature_matrix(matrices):
    #Input is tuple of matrices to stack
    total_mat = sc.sparse.hstack(matrices)
    return total_mat

def category_dummies(df):
    #create dummies for category_code
    dummies = pd.get_dummies(df['category_code'])
    dummies.drop('m4m', axis=1, inplace=True)    
    return dummies 

def random_forest(X_train, y_train): 
    rf = RandomForestRegressor(n_jobs = -1, random_state=42)
    rf.fit_transform(X_train, y_train) 
    return rf 

def gradient_boosting(X_train, y_train): 
    gb = GradientBoostingRegressor(n_jobs = -1, random_state=42)
    gb.fit_transform(X_train, y_train) 
    return gb 
    

if __name__=='__main__':	
    with open('training_data.pkl', 'rb') as fid:
	training_data = cPickle.load(fid)
    df = pd.concat(training_data[0], training_data[1], axis=1)
    df = df.ix[df['age'] < 50, :]
    target = df['age']
    #create tfidf matrix for total_text
    text_mat, text_features = tfidf_matrix(df['total_text'])
    #create tfidf matrix for titles 
    title_mat, title_features = tfidf_matrix(df['title'])
#    cat_dummies = category_dummies(df)
    #create list of features 
    total_features = []
    #total_features.extend(text_features)
    total_features.extend(title_features)
 #   total_features.extend(cat_dummies.columns.tolist())	
        
    #add total text length as feature
    df['total_text_length'] = df['total_text'].map(len)
    #combine matrices 
    total_mat = build_feature_matrix((text_mat, title_mat, np.array(df[['num_attributes', 'num_images', 'total_text_length']])))
    X_train, X_test, y_train, y_test = train_test_split(total_mat, target, test_size = 0.3)
	
    #random forest
    rf = random_forest(X_train, y_train)
    with open('rf_model_with_features.pkl', 'wb') as fid:
	cPickle.dump(rf, fid)
    print 'Random Forest Regressor R^2:'
    print rf.score(X_test, y_test)

