import modeling as md
import numpy as np 
import scipy as sc
import pandas as pd 
import cPickle 
import modeling as md
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def print_top_tokens(model, feature_names, n_top_words):
   for topic_idx, topic in enumerate(model.components_):
	print ("Topic #%d:" % topic_idx)
	print (' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words -1:-1]]))
	print ()

def get_data():
    with open('dataframe_for_eda.pkl', 'rb') as fid:
	df = cPickle.load(fid)
    return df

def reduce_dimensions(total_mat, n_topics):
    #input is data matrix, shape (n_samples, n_features)
    #returns W array, shape (n_samples, n_components)
    
    nmf = NMF(n_components = n_topics, random_state=42, alpha=0.1, l1_ratio = 0.5)
    nmf.fit(total_mat)
    return nmf 

if __name__=='__main__':
    df = get_data()
    #Use tfidf features for NMF
    text_mat, text_features = md.tfidf_matrix(df['total_text'])
    
    #Fit NMF
    n_samples = text_mat.shape[0]
    n_features = text_mat.shape[1]
    n_topics = 15
    n_top_words = 20
    print 'Fitting the NMF model with tf-idf features'
    nmf = reduce_dimensions(text_mat, n_topics) 
    print_top_tokens(nmf,text_features,n_top_words)

