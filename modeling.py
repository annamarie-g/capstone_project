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
from sklearn.svm import LinearSVC, SVC, SVR, LinearSVR

def create_collocations_from_trainingset(series):
    #calling for trainingset 
    scored_bigrams, scored_trigrams =tp.find_collocations(series)
    return scored_bigrams, scored_trigrams
 
def tfidf_matrix(series):
    with open('bigrams_MWEs.pkl', 'rb') as fid:
        bigrams = cPickle.load(fid) 

    vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 5, preprocessor = tp.custom_preprocessor, tokenizer = tp.custom_tokenizer, stop_words=stopwords.words('english'), lowercase=True)

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
    random_forest_grid = {'n_estimators':[x for x in range(150, 300, 50)], 'max_features': ['sqrt', 'log2', '2000']}
    rfr_gridsearch = GridSearchCV(RandomForestRegressor(), random_forest_grid, n_jobs = -1, verbose=True)
    rfr_gridsearch.fit(X_train, y_train)
    print "best random forest regressor model:"
    print rfr_gridsearch.best_params_
    return rfr_gridsearch.best_estimator_

def random_forest_classifier(X_train, y_train):
    random_forest_grid = {'n_estimators':[x for x in range(50, 400, 50)], 'max_features': ['sqrt', 'log2', 'auto', '250', '500', '1000', '2000']}
    rfc_gridsearch = GridSearchCV(RandomForestClassifier(), random_forest_grid, n_jobs = -1,  verbose=True)
    rfc_gridsearch.fit(X_train, y_train)
    print 'best random forest classifier model:'
    print rfc_gridsearch.best_params_
    return rfc_gridsearch.best_estimator_

def gradient_boosting(X_train, y_train): 
    gb = GradientBoostingRegressor(max_features = 'sqrt', random_state=42)
    gb.fit_transform(X_train, y_train) 
    return gb 

def svc_rbf(X_train, y_train_clf):
    est = SVC(kernel='rbf', random_state = 42)
    clf = GridSearchCV(est, parameters, cv=2, n_jobs=-1)
    clf.fit(X_train, y_train_clf)
    print 'Best SVC estimator:'
    print clf.best_estimator_
    print 'Best accuracy score:'
    print clf.best_score_ 
    return clf.best_estimator_

def svr_rbf(X_train, y_train):
    parameters = {'C': [0.0001, 0.1, 100], 'loss':['l1', 'l2']}
    est = SVR(kernel='rbf')
    reg = GridSearchCV(est, parameters, cv=2, n_jobs=-1)
    reg.fit(X_train, y_train)
    print 'Best SVR estimator:'
    print reg.best_estimator_
    print 'Best R^2  score:'
    print reg.best_score_ 
    return reg.best_estimator_

def linear_svc(X_train, y_train_clf):
    #linear support vector classification
    parameters = {'C': [0.0001, 0.1, 100], 'loss':['l1', 'l2']}
    est = LinearSVC()
    clf = GridSearchCV(est, parameters, cv=2, n_jobs = -1) 
    clf.fit(X_train, y_train_clf)
    print 'best model:'
    print clf.best_estimator_
    print 'best score:' 
    print clf.best_score_
    print 'grid scores:' 
    print clf.grid_scores_     
    return clf.best_estimator_

def linear_svr(X_train, y_train):
    #linear support vector classification
    parameters = {'C': [0.0001,.005, 0.1, 100], 'loss':['l1', 'l2']}
    est = LinearSVR()
    clf = GridSearchCV(est, parameters, cv=2, n_jobs = -1) 
    clf.fit(X_train, y_train)
    print 'best model:'
    print clf.best_estimator_
    print 'best score:' 
    print clf.best_score_
    print 'grid scores:' 
    print clf.grid_scores_     
    return clf.best_estimator_

def create_age_groups(series):
    #cut into age groups 
    age_group = pd.cut(series, range(5,95,10), right=False, labels = ["{0} - {1}".format(i, i+9) for i in range(5, 85, 10)])
    return age_group

def reduce_dimensions(total_mat, n_topics):
    #input is data matrix, shape (n_samples, n_features)
    #returns W array, shape (n_samples, n_components)
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit_transform(total_mat)
    return nmf

def get_data():
    with open('df_age_predict_edited.pkl', 'rb') as fid:
	df = cPickle.load(fid)
    #df = pd.concat([training_data[0], training_data[1]], axis=1)
    #df = df.ix[df['category_code'] == 'm4w', :]
    target = df.pop('age')
    return df, target
    

def create_featurespace(df):
    #create tfidf matrix for total_text
    text_mat, text_features = tfidf_matrix(df['total_text'])
    #create tfidf matrix for titles 
    title_mat, title_features = tfidf_matrix(df['title'])
    cat_dummies = category_dummies(df)
    #create list of features 
    total_features = []
    total_features.extend(text_features)
    total_features.extend(title_features)
    total_features.extend(cat_dummies.columns.tolist())	
    #add total text length as feature
    df['total_text_length'] = df['total_text'].map(len)
    #combine matrices 
    total_mat = build_feature_matrix((text_mat, title_mat, cat_dummies,  np.array(df[['num_attributes', 'num_images', 'total_text_length']])))
    return total_mat 	

if __name__=='__main__':	
    df, target = get_data()
    total_mat = create_featurespace(df)
#    total_mat = reduce_dimensions(total_mat, n_topics=10000)

    X_train, X_test, y_train, y_test = train_test_split(total_mat, target, test_size = 0.3)

    #create age group on y_train and y_test 
    y_train_clf = create_age_groups(y_train)
    y_test_clf = create_age_groups(y_test)	

    rfr = random_forest_regressor(X_train, y_train)
    rfr.transform(X_train, y_train)
    print "Best Random Forest Regressor R^2:"
    print rfr.score(X_test, y_test)


'''
    svm_reg = linear_svr(X_train, y_train)
    svm_reg.score(X_test, y_test)

    reg = svr_rbf(X_train, y_train)
    print 'Best svc classifier accuracy:' 
    print reg.score(X_test, y_test) 

    clf = svc_rbf(X_train, y_train_clf)
    print 'Best svc classifier accuracy:' 
    print clf.score(X_test, y_test_clf) 

    rfc = random_forest_classifier(X_train, y_train_clf)
    rfc.transform(X_train, y_train_clf)
    print "Best Random Forest Classifier Accuracy:"
    print rfc.score(X_test, y_test_clf)
	
    rfr = random_forest_regressor(X_train, y_train)
    rfr.transform(X_train, y_train)
    print "Best Random Forest Regressor R^2:"
    print rfr.score(X_test, y_test)

    svm_clf = linear_svc(X_train, y_train_clf)
    svm_clf.score(X_test, y_test_clf)

    gb = gradient_boosting(X_train.todense(), y_train)
    print 'Gradient Boosted Model:'
    print gb.score(X_train, y_train) 
'''
