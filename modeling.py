import numpy as np
import scipy as sc
import pandas as pd 
import cPickle
import text_preprocessing as tp
from nltk.corpus import stopwords 
from sklearn.cluster import KMeans 
from sklearn.decomposition import NMF 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import train_test_split 
from sklearn.grid_search import GridSearchCV


def create_collocations_from_trainingset(series):
    pass 
 
def tfidf_matrix(series):
    vectorizer = TfidfVectorizer(tokenizer = tp.custom_tokenizer, stop_words=stopwords.words('english'), lowercase=True)
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

def random_forest_regressor(X_train, y_train): 
    random_forest_grid = {'n_estimators':[x for x in range(50, 400, 50)], 'max_features': ['sqrt', 'log2', 'auto', '250', '500', '1000', '2000']}
    rfr_gridsearch = GridSearchCV(RandomForestRegressor(), random_forest_grid, n_jobs = -1, verbose=True)
    rfr_gridsearch.fit_transform(X_train, y_train)
    print "best random forest regressor model:"
    print rfr_gridsearch.best_params_
    return rfr_gridsearch.best_estimator

def random_forest_classifier(X_train, y_train):
    random_forest_grid = {'n_estimators':[x for x in range(50, 400, 50)], 'max_features': ['sqrt', 'log2', 'auto', '250', '500', '1000', '2000']}
    rfc_gridsearch = GridSearchCV(RandomForestClassifier(), random_forest_grid, n_jobs = -1,  verbose=True)
    rfc_gridsearch.fit_transform(X_train, y_train)
    print 'best random forest classifier model:'
    print rfc_gridsearch.best_params_
    return rfc_gridsearch.best_estimator

def gradient_boosting(X_train, y_train): 
    gb = GradientBoostingRegressor(n_jobs = -1, random_state=42)
    gb.fit_transform(X_train, y_train) 
    return gb 
   
def create_age_groups(series):
    #cut into age groups 
    age_group = pd.cut(series, range(5,95,10), right=False, labels = ["{0} - {1}".format(i, i+9) for i in range(5, 85, 10)])
    return age_group

if __name__=='__main__':	
    with open('df_age_predict.pkl', 'rb') as fid:
	df = cPickle.load(fid)
    #df = pd.concat([training_data[0], training_data[1]], axis=1)
    #df = df.ix[df['category_code'] == 'stp', :]
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

    #create age group on y_train and y_test 
    y_train_clf = create_age_groups(y_train)
    y_test_clf = create_age_groups(y_test)	
	
    rfc = random_forest_classifier(X_train, y_train_clf)
	
    rfr = random_forest_regressor(X_train, y_train)

